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
    result = runner.invoke(app, ["convert", "org/model", "--plan"])
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


def test_convert_requires_confirmation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from npu_model import convert

    calls: list[str] = []
    monkeypatch.setattr(convert, "plan", lambda options: _FakePlan(tmp_path))
    monkeypatch.setattr(convert, "run", lambda plan, progress: calls.append("run"))
    result = runner.invoke(app, ["convert", "org/model"], input="n\n")
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
    result = runner.invoke(app, ["convert", "org/model", "--yes", "--json"])
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
