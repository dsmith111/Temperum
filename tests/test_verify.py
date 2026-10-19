from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from npu_model.core.errors import NpuModelError
from npu_model.templates import from_jinja
from npu_model.verify import CONVERSATIONS, reference_dtype, template_check, verdict, verify

CHATML = from_jinja("<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>")


def _report(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"load": {"ok": True}, "prompts": [{"tokens": 5, "degenerate": False}],
                            "quality": {"perplexity": {"ratio": 1.05}, "needle": []}, "template": {"status": "match"}}
    base.update(overrides)
    return base


@pytest.mark.parametrize("report,level,expected", [
    (_report(), "standard", "passed"),
    (_report(load={"ok": False, "error": "QNN failed"}), "standard", "failed"),
    (_report(prompts=[{"tokens": 32, "degenerate": True}]), "quick", "failed"),
    (_report(prompts=[]), "quick", "failed"),
    (_report(quality={}), "quick", "passed"),
    (_report(quality={"perplexity": {"ratio": 1.3}}), "standard", "warning"),
    (_report(quality={"perplexity": {"ratio": 2.4}}), "standard", "failed"),
    (_report(quality={"skipped": "not enough free memory"}), "standard", "warning"),
    # GGUF measures perplexity without the source model; answers still need it
    (_report(quality={"skipped": "not enough free memory", "perplexity": {"ratio": 1.05}}), "standard", "warning"),
    (_report(quality={"skipped": "not enough free memory", "perplexity": {"ratio": 2.0}}), "standard", "failed"),
    (_report(quality={"perplexity": {"ratio": 1.0}, "needle": [
        {"input_len": 1800, "correct": False, "reference_correct": True}]}), "standard", "warning"),
    (_report(quality={"perplexity": {"ratio": 1.0}, "needle": [  # the source model missed it too
        {"input_len": 1800, "correct": False, "reference_correct": False}]}), "standard", "passed"),
    (_report(quality={"perplexity": {"ratio": 1.0}, "needle": [  # no source answer: judged on its own
        {"input_len": 1800, "correct": False, "reference_correct": None}]}), "standard", "warning"),
    (_report(template={"status": "mismatch"}), "full", "warning"),
])
def test_verdicts_keep_claims_separate(report: dict[str, Any], level: str, expected: str) -> None:
    status, reasons = verdict(report, level)
    assert status == expected
    assert bool(reasons) == (expected != "passed")


def test_reference_dtype_fits_memory() -> None:
    assert reference_dtype(500_000_000, 1.3, 15.4) == "float32"  # 2 GB within ~3.9 GB budget
    assert reference_dtype(1_500_000_000, 1.3, 15.4) == "bfloat16"
    assert reference_dtype(4_000_000_000, 1.3, 15.4) is None
    assert reference_dtype(None, None, None) == "float32"


def test_template_check_compares_with_the_model_rendering() -> None:
    expected = [CHATML.render(c) or "" for c in CONVERSATIONS]
    assert template_check(CHATML, expected, None)["status"] == "match"
    assert template_check(CHATML, ["<s>" + e for e in expected], "<s>")["status"] == "match"  # runtime adds BOS
    assert template_check(CHATML, ["<s>" + e for e in expected], None)["status"] == "mismatch"
    assert template_check(CHATML, [], None)["status"] == "not-checked"


class FakeScripts:
    """Stands in for reference.py (quantize env) and npu_check.py (verify env)."""

    def __init__(self, ratio: float = 1.04, load_ok: bool = True) -> None:
        self.ratio = ratio
        self.load_ok = load_ok
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> Any:
        self.calls.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        lines: list[str] = []
        if cmd[1].endswith("reference.py"):
            spec = json.loads(Path(cmd[cmd.index("--spec") + 1]).read_text(encoding="utf-8"))
            out.write_text(json.dumps({"prompts": [{"prompt": p, "input_ids": [1], "output_ids": [2], "output": "x"}
                                                   for p in spec["prompts"]],
                                       "max_new_tokens": spec["max_new_tokens"], "needle": [],
                                       "rendered": [CHATML.render(c) for c in spec["conversations"]]}),
                           encoding="utf-8")
        else:
            lines = ["@@progress prompt 1/1\n"]
            report: dict[str, Any] = {"load": {"ok": self.load_ok}}
            if self.load_ok:
                report.update({"prompts": [{"prompt": "p", "output": "Paris", "tokens": 3, "degenerate": False,
                                            "matching_prefix": 1, "reference_tokens": 1, "exact": True}],
                               "speed": {"median_tokens_per_s": 50.0}, "tokenizer": {"adds_bos": False},
                               "perplexity": {"ratio": self.ratio}, "needle": []})
            out.write_text(json.dumps(report), encoding="utf-8")

        class Proc:
            stdout = iter(lines)

            def wait(self) -> int:
                return 0

        return Proc()


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "genai_config.json").write_text(json.dumps({"model": {"context_length": 4096}}), encoding="utf-8")
    return bundle


def test_standard_verification_runs_reference_then_npu(tmp_path: Path) -> None:
    fake = FakeScripts()
    seen: list[str] = []
    report = verify(_bundle(tmp_path), tmp_path, CHATML, "standard", quantize_python=Path("q.exe"),
                    verify_python=Path("v.exe"), work=tmp_path / "work", heldout=tmp_path / "heldout.jsonl",
                    params=500_000_000, available_gb=4.0, total_gb=16.0, progress=seen.append, popen=fake)
    assert [c[0] for c in fake.calls] == ["q.exe", "v.exe"]
    assert "--reference" in fake.calls[1]
    assert report["verdict"] == "passed" and report["template"]["status"] == "match"
    assert report["npu_placement"]["status"] == "npu-stages-executed"
    assert report["quality"]["reference_dtype"] == "float32" and "prompt 1/1" in seen


def test_quick_verification_skips_the_reference(tmp_path: Path) -> None:
    fake = FakeScripts()
    report = verify(_bundle(tmp_path), tmp_path, CHATML, "quick", quantize_python=Path("q.exe"),
                    verify_python=Path("v.exe"), work=tmp_path / "work", heldout=None, params=None,
                    available_gb=None, popen=fake)
    assert [c[0] for c in fake.calls] == ["v.exe"] and "--prompts" in fake.calls[0]
    formatted = json.loads(fake.calls[0][fake.calls[0].index("--prompts") + 1])
    assert formatted[0].startswith("<|im_start|>system\nYou are helpful.")
    assert report["verdict"] == "passed" and report["quality"] == {}


def test_load_failure_is_a_failed_verdict(tmp_path: Path) -> None:
    report = verify(_bundle(tmp_path), tmp_path, CHATML, "quick", quantize_python=None, verify_python=Path("v.exe"),
                    work=tmp_path / "work", heldout=None, params=None, available_gb=None,
                    popen=FakeScripts(load_ok=False))
    assert report["verdict"] == "failed" and report["npu_placement"]["status"] == "not-established"


def test_none_level_and_unknown_levels(tmp_path: Path) -> None:
    report = verify(_bundle(tmp_path), tmp_path, CHATML, "none", quantize_python=None, verify_python=Path(),
                    work=tmp_path, heldout=None, params=None, available_gb=None)
    assert report["verdict"] == "not-verified"
    with pytest.raises(NpuModelError):
        verify(_bundle(tmp_path / "x"), tmp_path, CHATML, "thorough", quantize_python=None, verify_python=Path(),
               work=tmp_path, heldout=None, params=None, available_gb=None)
