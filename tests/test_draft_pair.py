"""A speculative-decoding draft paired with its target: the plan, the bundle, the paired verification, the flows."""
from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from temperum import flows, gguf, gguf_verify
from temperum.cli.main import app
from temperum.core.errors import TemperumError
from temperum.host import HostInfo
from temperum.profiles import get_target
from temperum.sources import ModelRef, ResolvedModel
from temperum.templates import ChatTemplate
from temperum.toolchains.llama_cpp import ConverterSource

X_ELITE = HostInfo(system="Windows", native_arch="arm64", python_platform="win-amd64",
                   processor_name="Snapdragon(R) X Elite - X1E78100 - Qualcomm(R) Oryon(TM) CPU",
                   total_ram_gb=15.4, available_ram_gb=6.0)
TARGET_CONFIG = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "torch_dtype": "bfloat16",
                 "max_position_embeddings": 40960, "num_hidden_layers": 36, "hidden_size": 2560}
DRAFT_CONFIG = {"architectures": ["DFlashDraftModel"], "model_type": "qwen3", "dtype": "bfloat16", "block_size": 16,
                "dflash_config": {"target_layer_ids": [1, 9, 17, 25, 33]}, "num_target_layers": 36,
                "hidden_size": 2560, "num_hidden_layers": 5}


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEMPERUM_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("temperum.convert.disk_free_gb", lambda _path: 500.0)  # a 4B pair needs 14 GB of disk


def _target() -> ResolvedModel:
    return ResolvedModel(ModelRef("hf", repo="Qwen/Qwen3-4B"), commit="1cfa9a720891", gated=False, license="apache-2.0",
                         weight_bytes=8_060_000_000, config=dict(TARGET_CONFIG))


def _draft(config: dict[str, Any] | None = None, reason: str | None = "architecture DFlashDraftModel",
           target: str | None = "Qwen/Qwen3-4B") -> ResolvedModel:
    return ResolvedModel(ModelRef("hf", repo="z-lab/Qwen3-4B-DFlash-b16"), commit="b74e3a329c", gated=False,
                         license="mit", weight_bytes=1_074_860_568, config=dict(config or DRAFT_CONFIG),
                         draft_reason=reason, draft_target=target if reason else None)


def _payload(root: Path) -> Path:
    lib = root / "lib" / "ollama"
    (lib / "hexagon").mkdir(parents=True)
    exe = root / "ollama.exe"
    exe.write_bytes(b"")
    for name in ("llama-quantize.exe", "llama-server.exe", "llama-imatrix.exe", "llama-quantize", "llama-server",
                 "llama-imatrix"):
        (lib / name).write_bytes(b"")
    (lib / "hexagon" / "ggml-hexagon.dll").write_bytes(b"")
    for name in gguf.HTP_LIBRARIES:
        (lib / "hexagon" / name).write_bytes(b"")
    return exe


def _run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, "version: 0.6.0-dev (build 1, commit e467e5b)", "")


def test_a_draft_alone_stops_the_plan_with_the_pairing_command(tmp_path: Path) -> None:
    with pytest.raises(TemperumError) as caught:
        gguf.plan(_draft(), X_ELITE, get_target("gguf-hexagon"), ollama_exe=_payload(tmp_path), run=_run,
                  out=tmp_path / "a")
    assert caught.value.reason_code == "DRAFT_NEEDS_TARGET" and "only runs paired" in str(caught.value)
    assert caught.value.hint == "Convert the pair: temperum convert Qwen/Qwen3-4B --draft z-lab/Qwen3-4B-DFlash-b16"
    with pytest.raises(TemperumError) as caught:
        gguf.plan(_draft(reason="tagged draft-model on Hugging Face", target=None), X_ELITE,
                  get_target("gguf-hexagon"), ollama_exe=_payload(tmp_path / "b"), run=_run, out=tmp_path / "c")
    assert caught.value.hint and "<target model>" in caught.value.hint and "card names it" in caught.value.hint


def test_a_pair_plans_with_the_drafts_block_and_its_own_job(tmp_path: Path) -> None:
    exe = _payload(tmp_path)
    target = get_target("gguf-hexagon")
    plain = gguf.plan(_target(), X_ELITE, target, ollama_exe=exe, run=_run, out=tmp_path / "a")
    pair = gguf.plan(_target(), X_ELITE, target, ollama_exe=exe, run=_run, out=tmp_path / "b", draft=_draft())
    assert pair.draft is not None and pair.draft.ref.repo == "z-lab/Qwen3-4B-DFlash-b16" and pair.draft_block == 16
    assert plain.draft is None and plain.draft_block == 0 and plain.to_dict()["draft"] is None
    assert pair.to_dict()["draft"]["repo"] == "z-lab/Qwen3-4B-DFlash-b16" and pair.to_dict()["draft_block"] == 16
    assert pair.job_id != plain.job_id and pair.estimates["minutes"] > plain.estimates["minutes"]
    assert pair.estimates["download_gb"] > plain.estimates["download_gb"]
    assert any("guesses are accepted" in w for w in pair.warnings) and not any("guesses" in w for w in plain.warnings)
    console = Console(file=io.StringIO(), width=220)
    console.print(flows.gguf_plan_table(pair))
    text = console.file.getvalue()  # type: ignore[attr-defined]
    assert "Draft" in text and "z-lab/Qwen3-4B-DFlash-b16" in text and "16 tokens per step" in text


@pytest.mark.parametrize("draft,code,words", [
    (_draft(reason=None), "DRAFT_NOT_A_DRAFT", "is not a speculative-decoding draft"),
    (_draft({"architectures": ["LlamaForCausalLMEagle3"], "draft_vocab_size": 1, "block_size": 4},
            "config has draft_vocab_size"), "DRAFT_NOT_PAIRABLE", "would launch this one as an MTP draft"),
    (_draft({**DRAFT_CONFIG, "block_size": None, "dflash_config": {}}), "DRAFT_NO_BLOCK", "no block_size"),
    (_draft({**DRAFT_CONFIG, "num_target_layers": 48}), "DRAFT_MISMATCH", "target with 48 layers"),
    (_draft({**DRAFT_CONFIG, "model_type": "llama"}), "DRAFT_MISMATCH", "is a llama draft"),
])
def test_the_pair_is_checked_before_anything_is_downloaded(tmp_path: Path, draft: ResolvedModel, code: str,
                                                            words: str) -> None:
    with pytest.raises(TemperumError) as caught:
        gguf.plan(_target(), X_ELITE, get_target("gguf-hexagon"), ollama_exe=_payload(tmp_path), run=_run,
                  out=tmp_path / "a", draft=draft)
    assert caught.value.reason_code == code and words in str(caught.value)


def test_the_modelfile_pairs_the_draft_and_the_template_rewrite_keeps_it(tmp_path: Path) -> None:
    record = {"file": "draft.gguf", "block_size": 16}
    lines = gguf.modelfile(4096, draft=record).splitlines()
    assert lines[:4] == ["FROM ./model.gguf", "DRAFT ./draft.gguf", "PARAMETER draft_num_predict 16",
                         "PARAMETER num_ctx 4096"]
    assert "DRAFT" not in gguf.modelfile(4096)

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "LICENSE").write_text("mit", encoding="utf-8")
    model = tmp_path / "model-Q4_0.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 64)
    draft_file = tmp_path / "draft-Q8_0.gguf"
    draft_file.write_bytes(b"GGUF" + b"\1" * 32)
    plan = SimpleNamespace(destination=tmp_path / "bundle", context=4096, quant="Q4_0", model=_target(),
                           target=get_target("gguf-hexagon"), runtime=gguf.LLAMA_RUNTIMES[gguf.DEFAULT_LLAMA_RUNTIME])
    bundle = gguf.assemble(model, snapshot, plan, {"converter_source": "e467e5b"},  # type: ignore[arg-type]
                           draft=(draft_file, {"repo": "z-lab/Qwen3-4B-DFlash-b16", "commit": "b74e3a329c",
                                               "architecture": "DFlashDraftModel", "block_size": 16,
                                               "precision": "Q8_0"}))
    assert (bundle / "draft.gguf").read_bytes() == draft_file.read_bytes()
    manifest = json.loads((bundle / gguf.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["draft"]["file"] == "draft.gguf" and manifest["draft"]["block_size"] == 16
    assert manifest["draft"]["repo"] == "z-lab/Qwen3-4B-DFlash-b16" and "draft.gguf" in manifest["files"]
    assert "DRAFT ./draft.gguf" in (bundle / "Modelfile").read_text(encoding="utf-8")
    chat = ChatTemplate("chatml", "{{ .Prompt }}", ("<|im_end|>",), None, True)
    gguf.set_template(bundle, chat, 4096)
    text = (bundle / "Modelfile").read_text(encoding="utf-8")
    assert "DRAFT ./draft.gguf" in text and "PARAMETER draft_num_predict 16" in text and "TEMPLATE" in text


def test_convert_draft_hands_the_converter_the_targets_snapshot(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class Proc:
        returncode = 0
        stdout = iter(["INFO:hf-to-gguf:Model successfully exported\n"])

        def wait(self) -> int:
            return 0

    def popen(cmd: list[str], **_kwargs: Any) -> Proc:
        seen["cmd"] = cmd
        Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"GGUF")
        return Proc()

    source = ConverterSource("b11644", "e467e5b", "https://example/archive", {})
    out = gguf.convert_draft(Path("python"), source, tmp_path / "draft", tmp_path / "target", tmp_path / "draft-Q8_0.gguf",
                             tmp_path / "log", lambda _m: None, popen=popen)
    cmd = seen["cmd"]
    assert out.is_file() and cmd[2] == str(tmp_path / "draft")
    assert cmd[cmd.index("--target-model-dir") + 1] == str(tmp_path / "target")
    assert cmd[cmd.index("--outtype") + 1] == "q8_0"


def _verify_with_fake_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, draft_fails: bool = False) -> dict[str, Any]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "draft.gguf").write_bytes(b"GGUF")
    served: list[list[str]] = []

    class FakeServer:
        def __init__(self, *_args: Any) -> None:
            self.extra_args: list[str] = []

        def __enter__(self) -> FakeServer:
            served.append(list(self.extra_args))
            if self.extra_args and draft_fails:
                raise RuntimeError("llama-server exited with code 1: dflash requires ctx_other")
            return self

        def __exit__(self, *_exc: object) -> None: ...

        def text(self) -> str:
            size = "2819.1" if self.extra_args else "2274.4"
            return ("load_tensors: offloaded 37/37 layers to GPU\n"
                    f"load_tensors:         HTP0 model buffer size =   {size} MiB\n")

        def bos_text(self) -> str | None:
            return None

        def apply_template(self, messages: list[dict[str, str]]) -> str:
            return f"<user>{messages[-1]['content']}<assistant>"

        def complete(self, prompt: Any, n_predict: int) -> dict[str, Any]:
            if self.extra_args:  # the pair: faster on the first prompt, slower on the others, same answer
                first = isinstance(prompt, str) and prompt.startswith("<user>What is the capital")
                return {"content": "Paris", "tokens": [1, 2, 3],
                        "timings": {"predicted_per_second": 28.0 if first else 9.0, "draft_n": 48,
                                    "draft_n_accepted": 22 if first else 5}}
            return {"content": "Paris", "tokens": [1, 2, 3], "timings": {"predicted_per_second": 16.0}}

    monkeypatch.setattr(gguf_verify, "LlamaServer", FakeServer)
    monkeypatch.setattr(gguf_verify, "run_script", lambda *_a, **_k: (1, "no tokenizer"))
    plan = SimpleNamespace(verify_level="quick", payload=None, context=4096, estimates={"params_billions": 4.0},
                           host=SimpleNamespace(available_ram_gb=3.0, total_ram_gb=16.0), draft=_draft(), draft_block=16)
    report = gguf_verify.verify_gguf(bundle, tmp_path / "source", plan, SimpleNamespace(python_exe=lambda _r: Path("py")),  # type: ignore[arg-type]
                                     tmp_path / "work", lambda _m: None)
    report["_served"] = served
    return report


def test_paired_verification_measures_the_draft_against_plain_decoding(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    report = _verify_with_fake_server(tmp_path, monkeypatch)
    pair = report["_served"][-1]
    assert pair[:2] == ["--spec-type", "draft-dflash"] and pair[pair.index("--spec-draft-n-max") + 1] == "16"
    assert str(tmp_path / "bundle" / "draft.gguf") in pair
    draft = report["draft"]
    assert draft["loaded"] and draft["placement"]["htp_weight_mib"] == 2819.1 and draft["block_size"] == 16
    speedups = [p["speedup"] for p in draft["prompts"]]
    assert speedups[0] == 1.75 and all(s == 0.56 for s in speedups[1:]) and draft["speedup_range"] == [0.56, 1.75]
    assert draft["prompts"][0]["acceptance"] == 0.458 and draft["acceptance_range"] == [0.104, 0.458]
    assert all(p["same_answer"] for p in draft["prompts"])
    assert report["verdict"] == "passed"
    line = report["reasons"][-1]
    assert line.startswith("draft on the NPU beside the target: decode 0.56x to 1.75x of plain")
    assert "10% to 46% of its guesses accepted" in line and "draft_num_predict 0 turns it off" in line
    summary = flows.gguf_summary(SimpleNamespace(bundle=tmp_path, job_dir=tmp_path, verification=report, recipe=None,
                                                 calibration=None, draft={"repo": "z-lab/x", "file": "draft.gguf"}),
                                 start=0.0)
    assert any("draft on the NPU beside the target" in text for text in flows.ready_lines(summary))


def test_a_draft_that_does_not_load_fails_the_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _verify_with_fake_server(tmp_path, monkeypatch, draft_fails=True)
    assert report["draft"]["loaded"] is False and "ctx_other" in report["draft"]["error"]
    assert report["verdict"] == "failed" and report["reasons"][0].startswith("the draft did not load beside the target")


def test_plan_gguf_asks_for_the_target_of_a_draft_and_pairs_them(monkeypatch: pytest.MonkeyPatch) -> None:
    models = {"z-lab/Qwen3-4B-DFlash-b16": _draft(), "Qwen/Qwen3-4B": _target()}
    monkeypatch.setattr("temperum.sources.resolve", lambda ref, **_k: models[str(ref.repo)])
    monkeypatch.setattr("temperum.host.detect_host", lambda: X_ELITE)
    planned: list[dict[str, Any]] = []

    def fake_plan(model: Any, host: Any, target: Any, **kwargs: Any) -> Any:
        planned.append({"model": model.ref.repo, "draft": kwargs["draft"].ref.repo if kwargs["draft"] else None})
        return SimpleNamespace(model=model)

    monkeypatch.setattr("temperum.gguf.plan", fake_plan)
    asked: list[tuple[str, str]] = []

    def ask(question: str, default: str) -> str:
        asked.append((question, default))
        return default

    common = dict(target_id="gguf-hexagon", quant="Q4_0", context=None, out_dir=None, verify_level="standard",
                  overwrite=False, python=None, machine=X_ELITE)
    flows.plan_gguf("z-lab/Qwen3-4B-DFlash-b16", ask=ask, **common)  # type: ignore[arg-type]
    assert asked[0][1] == "Qwen/Qwen3-4B" and "only runs paired" in asked[0][0]
    flows.plan_gguf("z-lab/Qwen3-4B-DFlash-b16", target_model="Qwen/Qwen3-4B", ask=ask, **common)  # type: ignore[arg-type]
    flows.plan_gguf("Qwen/Qwen3-4B", draft="z-lab/Qwen3-4B-DFlash-b16", **common)  # type: ignore[arg-type]
    flows.plan_gguf("Qwen/Qwen3-4B", **common)  # type: ignore[arg-type]
    assert planned == [{"model": "Qwen/Qwen3-4B", "draft": "z-lab/Qwen3-4B-DFlash-b16"}] * 3 + \
        [{"model": "Qwen/Qwen3-4B", "draft": None}]
    assert len(asked) == 1  # --target-model skips the question
    with pytest.raises(TemperumError) as caught:  # an empty answer backs out
        flows.plan_gguf("z-lab/Qwen3-4B-DFlash-b16", ask=lambda _q, _d: "", **common)  # type: ignore[arg-type]
    assert caught.value.reason_code == "DRAFT_NEEDS_TARGET" and caught.value.hint and "--draft" in caught.value.hint
    with pytest.raises(TemperumError):  # no way to ask (--yes, --json): the plan stops with the command
        flows.plan_gguf("z-lab/Qwen3-4B-DFlash-b16", **common)  # type: ignore[arg-type]


def test_convert_takes_draft_and_target_model(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr("temperum.cli.main._convert_gguf", lambda model, target, **kwargs: seen.append(kwargs))
    runner = CliRunner()
    assert runner.invoke(app, ["convert", "Qwen/Qwen3-4B", "--plan", "--target", "gguf-hexagon",
                               "--draft", "z-lab/Qwen3-4B-DFlash-b16"]).exit_code == 0
    assert runner.invoke(app, ["convert", "z-lab/Qwen3-4B-DFlash-b16", "--plan", "--target", "gguf-hexagon",
                               "--target-model", "Qwen/Qwen3-4B"]).exit_code == 0
    assert [(k["draft"], k["target_model"]) for k in seen] == [("z-lab/Qwen3-4B-DFlash-b16", None),
                                                              (None, "Qwen/Qwen3-4B")]


def test_the_interface_parses_draft_options() -> None:
    from temperum.tui.app import _options

    positional, opts = _options(["Qwen/Qwen3-4B", "--draft", "z-lab/Qwen3-4B-DFlash-b16"], set(),
                                {"--draft", "--target-model"})
    assert positional == ["Qwen/Qwen3-4B"] and opts == {"--draft": "z-lab/Qwen3-4B-DFlash-b16"}
