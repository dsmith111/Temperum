from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from npu_model.cli.main import app
from npu_model.core.errors import NpuModelError

runner = CliRunner()


def test_models_lists_recipes_families_and_verified_conversions() -> None:
    result = runner.invoke(app, ["models", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "Qwen/Qwen2.5-0.5B-Instruct" in data["curated"] and data["families"]["llama"]
    assert any(entry["repo"] == "Qwen/Qwen2.5-0.5B-Instruct" for entry in data["verified"])


def test_convert_plan_errors_are_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def gated(_options: Any) -> Any:
        raise NpuModelError("source", "HF_GATED", "org/model is gated", hint="Request access at https://hf.co/org/model")

    monkeypatch.setattr("npu_model.convert.plan", gated)
    result = runner.invoke(app, ["convert", "org/model", "--plan", "--target", "qnn-x-elite"])
    assert result.exit_code == 2
    assert "HF_GATED" in result.output and "Request access" in result.output


def test_install_options_are_checked_before_converting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    planned: list[str] = []
    monkeypatch.setattr("npu_model.convert.plan", lambda options: planned.append("plan"))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("NPU_MODEL_OLLAMA_EXE", raising=False)
    result = runner.invoke(app, ["convert", "org/model", "--install"])
    assert result.exit_code == 2 and "HOST_REQUIRED" in result.output
    result = runner.invoke(app, ["convert", "org/model", "--install", "--host", "http://127.0.0.1:11434",
                                 "--ollama-exe", str(tmp_path / "missing.exe")])
    assert result.exit_code == 2 and "EXE_REQUIRED" in result.output
    assert planned == []


def test_auto_target_takes_the_gguf_route_and_qnn_is_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    from npu_model.host import HostInfo

    machine = HostInfo(system="Windows", native_arch="arm64", python_platform="win-arm64",
                       processor_name="Snapdragon(R) X Elite - X1E78100 - Qualcomm(R) Oryon(TM) CPU")
    routes: list[tuple[str, Any]] = []

    def qnn_plan(options: Any) -> Any:
        routes.append(("qnn", options.target))
        raise NpuModelError("plan", "STOP", "stop here")

    monkeypatch.setattr("npu_model.host.detect_host", lambda: machine)
    monkeypatch.setattr("npu_model.cli.main._convert_gguf",
                        lambda model, target, **kwargs: routes.append((target, kwargs["machine"])))
    monkeypatch.setattr("npu_model.convert.plan", qnn_plan)
    for switch in ("1", "0"):  # the server switch no longer picks the target (D14)
        monkeypatch.setenv("OLLAMA_HEXAGON", switch)
        assert runner.invoke(app, ["convert", "org/model", "--plan"]).exit_code == 0
    assert runner.invoke(app, ["convert", "org/model", "--plan", "--target", "qnn-x-elite"]).exit_code == 2
    monkeypatch.setenv("NPU_MODEL_TARGET", "qnn-x-elite")  # or set once
    assert runner.invoke(app, ["convert", "org/model", "--plan"]).exit_code == 2
    assert routes == [("gguf-hexagon", machine), ("gguf-hexagon", machine), ("qnn", "qnn-x-elite"),
                      ("qnn", "qnn-x-elite")]


def test_convert_requires_confirmation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from npu_model import convert

    calls: list[str] = []
    monkeypatch.setattr(convert, "plan", lambda options: _FakePlan(tmp_path))
    monkeypatch.setattr(convert, "run", lambda plan, progress: calls.append("run"))
    result = runner.invoke(app, ["convert", "org/model", "--target", "qnn-x-elite"], input="n\n")
    assert result.exit_code == 1 and calls == []


def test_json_mode_streams_events_and_prints_the_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from npu_model import convert

    def fake_run(plan: Any, progress: Any) -> Any:
        progress("pass sq:onnxstaticquantization")
        return SimpleNamespace(bundle=tmp_path / "bundle", job_dir=tmp_path / "job", verification={
            "verdict": "passed", "reasons": [], "quality": {"perplexity": {"ratio": 1.04}},
            "speed": {"median_tokens_per_s": 50.0}})

    monkeypatch.setattr(convert, "plan", lambda options: _FakePlan(tmp_path))
    monkeypatch.setattr(convert, "run", fake_run)
    result = runner.invoke(app, ["convert", "org/model", "--yes", "--json", "--target", "qnn-x-elite"])
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.stderr.splitlines() if line.startswith("{")]
    assert [e["event"] for e in events] == ["plan", "progress"]
    assert events[1]["message"] == "pass sq:onnxstaticquantization"
    summary = json.loads(result.stdout)
    assert summary["verdict"] == "passed" and summary["perplexity_ratio"] == 1.04


class _FakePlan:
    def __init__(self, tmp_path: Path) -> None:
        from npu_model.profiles import get_runtime, get_target
        from npu_model.recipes import catalog, choose
        from npu_model.sources import ModelRef, ResolvedModel

        self.model = ResolvedModel(ModelRef("hf", repo="org/model"), commit="a" * 40, gated=False,
                                   license="apache-2.0", config={"model_type": "qwen2"}, weight_bytes=1)
        self.recipe = choose("Qwen/Qwen2.5-0.5B-Instruct", "qwen2", "Qwen2ForCausalLM")
        assert self.recipe.recipe in catalog().values()
        self.target, self.runtime = get_target("qnn-x-elite"), get_runtime("fork-qnn-1.24.1")
        self.quality, self.device, self.verify_level = "standard", "cpu", "standard"
        self.destination = tmp_path / "bundle"
        self.estimates = {"params_billions": 0.5, "minutes": 15, "ram_gb": 5.0, "disk_gb": 7.0, "download_gb": 1.0}
        self.warnings = ["a warning"]

    def to_dict(self) -> dict[str, Any]:
        return {"model": "org/model", "warnings": self.warnings}


def test_setup_prepares_what_the_target_needs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from npu_model.host import HostInfo

    machine = HostInfo(system="Windows", native_arch="arm64", python_platform="win-arm64",
                       processor_name="Snapdragon(R) X Elite - X1E78100 - Qualcomm(R) Oryon(TM) CPU")
    built: list[str] = []
    fetched: list[str] = []

    class FakeChains:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None: ...
        def ensure(self, role: str) -> Any:
            built.append(role)
            return SimpleNamespace(state="ready", detail="", path=tmp_path / role, tested="yes")

    class FakeSource:
        tag, directory = "b11644", tmp_path / "llama.cpp"
        @classmethod
        def load(cls, _tag: str) -> FakeSource: return cls()
        def ensure(self, **_kwargs: Any) -> Path:
            fetched.append(self.tag)
            return self.directory
        def problems(self) -> list[str]: return []

    monkeypatch.setattr("npu_model.host.detect_host", lambda: machine)
    monkeypatch.setattr("npu_model.toolchains.Toolchains", FakeChains)
    monkeypatch.setattr("npu_model.toolchains.llama_cpp.ConverterSource", FakeSource)
    # The default (a GGUF) needs the conversion toolchain and llama.cpp's converter, not the QNN toolchains.
    assert runner.invoke(app, ["setup"]).exit_code == 0
    assert built == ["quantize"] and fetched == ["b11644"]
    built.clear()
    assert runner.invoke(app, ["setup", "--target", "qnn-x-elite"]).exit_code == 0
    assert built == ["quantize", "compile", "verify"] and fetched == ["b11644"]


def test_doctor_names_the_missing_fork_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    from npu_model.cli.main import _fork_status

    monkeypatch.delenv("NPU_MODEL_OLLAMA_EXE", raising=False)
    assert _fork_status()["state"] == "not set" and "NPU_MODEL_OLLAMA_EXE" in _fork_status()["hint"]
    monkeypatch.setenv("NPU_MODEL_OLLAMA_EXE", "C:/nowhere/ollama.exe")
    assert _fork_status()["state"] == "missing"
