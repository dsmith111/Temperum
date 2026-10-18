from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from npu_model.olive_run import diagnose, olive_env, output_complete, run_olive

CRASH_LOG = """[INFO] [engine.py:665:_run_pass] Running pass st:staticllm
[INFO] [engine.py:665:_run_pass] Running pass cb:epcontextbinarygenerator
Traceback (most recent call last):
subprocess.CalledProcessError: Command '['python.EXE', 'pass_runner.py']' returned non-zero exit status 3221225477.
[WARNING] [engine.py:237:run] No output model produced. Please check the log for details.
"""
MISSING_DEP_LOG = """[INFO] [engine.py:665:_run_pass] Running pass cb:epcontextbinarygenerator
Stderr: Traceback (most recent call last):
ModuleNotFoundError: No module named 'requests'
subprocess.CalledProcessError: Command '['python.EXE', 'pass_runner.py']' returned non-zero exit status 1.
"""
GENAI_CONFIG = {"model": {"type": "decoder-pipeline", "decoder": {"pipeline": [
    {"embeddings": {"filename": "embeddings.onnx"}}, {"context_ctx": {"filename": "context_ctx.onnx"}},
    {"lm_head": {"filename": "lm_head.onnx"}}]}}}


def _write_bundle(out: Path, complete: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "genai_config.json").write_text(json.dumps(GENAI_CONFIG), encoding="utf-8")
    for name in ("embeddings.onnx", "context_ctx.onnx", "lm_head.onnx")[: 3 if complete else 2]:
        (out / name).write_bytes(b"x")


class FakeOlive:
    """One scripted outcome per attempt: "ok", "crash" or "missing-dep"."""

    def __init__(self, outcomes: list[str]) -> None:
        self.outcomes = outcomes
        self.commands: list[list[str]] = []
        self.envs: list[dict[str, str]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> Any:
        self.commands.append(cmd)
        self.envs.append(kwargs["env"])
        outcome = self.outcomes[len(self.commands) - 1]
        config = json.loads(Path(cmd[cmd.index("--config") + 1]).read_text(encoding="utf-8"))
        log = Path(cmd[cmd.index("--log") + 1])
        if outcome == "ok":
            _write_bundle(Path(config["output_dir"]))
            log.write_text("[INFO] Running pass cp:composeonnxmodels\n", encoding="utf-8")
        else:
            log.write_text(CRASH_LOG if outcome == "crash" else MISSING_DEP_LOG, encoding="utf-8")
        summary = {"exit_code": 0, "seconds": 10.0, "peak_rss_gb": 4.5 if outcome == "ok" else 3.0,
                   "passes": [{"pass": "cb:epcontextbinarygenerator", "start": 1.0, "seconds": 9.0}]}

        class Proc:
            stdout = iter(["@@pass cb:epcontextbinarygenerator 1.0\n", "@@summary " + json.dumps(summary) + "\n"])

            def wait(self) -> int:
                return 0

        return Proc()


def _config(tmp_path: Path) -> dict[str, Any]:
    return {"passes": {}, "output_dir": str(tmp_path / "out")}


def test_successful_run_reports_timing_and_memory(tmp_path: Path) -> None:
    fake = FakeOlive(["ok"])
    seen: list[str] = []
    result = run_olive(Path("python.exe"), _config(tmp_path), tmp_path / "work", progress=seen.append, popen=fake)
    assert result.ok and result.attempts == 1 and result.peak_rss_gb == 4.5 and result.seconds == 10.0
    assert seen == ["pass cb:epcontextbinarygenerator"]
    assert json.loads((tmp_path / "work" / "olive_config.json").read_text(encoding="utf-8"))["output_dir"]
    env = fake.envs[0]
    assert env["OLIVE_DISABLE_TELEMETRY"] == "1" and env["HF_HUB_OFFLINE"] == "1"


def test_compile_crash_is_retried_from_the_cache(tmp_path: Path) -> None:
    fake = FakeOlive(["crash", "ok"])
    seen: list[str] = []
    result = run_olive(Path("python.exe"), _config(tmp_path), tmp_path / "work", progress=seen.append, popen=fake)
    assert result.ok and result.attempts == 2 and len(result.logs) == 2
    assert any("retrying (2/3)" in s for s in seen)


def test_deterministic_failures_are_not_retried(tmp_path: Path) -> None:
    fake = FakeOlive(["missing-dep", "ok"])
    result = run_olive(Path("python.exe"), _config(tmp_path), tmp_path / "work", popen=fake)
    assert not result.ok and result.attempts == 1
    assert result.failed_pass == "cb:epcontextbinarygenerator"
    assert result.error == "ModuleNotFoundError: No module named 'requests'"


def test_a_rerun_keeps_the_logs_of_the_interrupted_run(tmp_path: Path) -> None:
    run_olive(Path("python.exe"), _config(tmp_path), tmp_path / "work", popen=FakeOlive(["missing-dep"]))
    result = run_olive(Path("python.exe"), _config(tmp_path), tmp_path / "work", popen=FakeOlive(["ok"]))
    assert result.ok and result.logs == [str(tmp_path / "work" / "olive-2.log")]
    assert (tmp_path / "work" / "olive-1.log").is_file()


def test_repeated_crashes_give_up_after_the_last_attempt(tmp_path: Path) -> None:
    result = run_olive(Path("python.exe"), _config(tmp_path), tmp_path / "work", popen=FakeOlive(["crash"] * 3))
    assert not result.ok and result.attempts == 3 and "status 3221225477" in (result.error or "")


def test_diagnose_and_completeness(tmp_path: Path) -> None:
    assert diagnose(CRASH_LOG)[2] is True and diagnose(MISSING_DEP_LOG)[2] is False
    assert diagnose("")[0] is None
    _write_bundle(tmp_path / "partial", complete=False)
    assert not output_complete(tmp_path / "partial") and not output_complete(tmp_path / "absent")
    _write_bundle(tmp_path / "full")
    assert output_complete(tmp_path / "full")


def test_olive_env_drops_credentials(monkeypatch: Any) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    assert "HF_TOKEN" not in olive_env()
