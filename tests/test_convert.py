from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from npu_model.convert import ConvertOptions, Job, estimates, plan
from npu_model.core.errors import NpuModelError
from npu_model.host import HostInfo
from npu_model.sources import ModelRef, ResolvedModel
from npu_model.toolchains import EnvStatus

X_ELITE = HostInfo(system="Windows", native_arch="arm64", python_platform="win-amd64",
                   processor_name="Snapdragon(R) X Elite - X1E78100 - Qualcomm(R) Oryon(TM) CPU",
                   total_ram_gb=15.4, available_ram_gb=6.0)
COMMIT = "7ae557604adf67be50417f59c2c2f167def9a775"


def _model(repo: str = "Qwen/Qwen2.5-0.5B-Instruct", model_type: str = "qwen2", arch: str = "Qwen2ForCausalLM",
           weight_bytes: int = 988_000_000, license_id: str = "apache-2.0") -> ResolvedModel:
    return ResolvedModel(ModelRef("hf", repo=repo), commit=COMMIT, gated=False, license=license_id,
                         config={"model_type": model_type, "architectures": [arch], "torch_dtype": "bfloat16"},
                         weight_bytes=weight_bytes)


class FakeChains:
    def __init__(self, host: HostInfo, runtime: Any, device: str, **kwargs: Any) -> None:
        self.device = device

    def status(self, role: str) -> EnvStatus:
        return EnvStatus(role, Path(role), "ready" if role != "verify" else "missing", tested="M0")


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPU_MODEL_HOME", str(tmp_path / "home"))


def _plan(options: ConvertOptions, host: HostInfo = X_ELITE, model: ResolvedModel | None = None) -> Any:
    return plan(options, host, resolver=lambda _ref: model or _model(), chains_factory=FakeChains)


def test_curated_model_plan_on_this_machine(tmp_path: Path) -> None:
    result = _plan(ConvertOptions(model="Qwen/Qwen2.5-0.5B-Instruct"))
    assert (result.target.id, result.quality, result.device, result.verify_level) == (
        "qnn-x-elite", "standard", "cpu", "standard")
    assert result.recipe.kind == "curated"
    assert result.destination == tmp_path / "home" / "models" / "qwen2.5-0.5b-instruct-qnn-x-elite"
    assert result.job_dir.parent == tmp_path / "home" / "jobs" and len(result.job_id) == 12
    assert any("first use: sets up verify" in w for w in result.warnings)
    assert not any("licence" in w for w in result.warnings)


def test_job_id_changes_with_inputs() -> None:
    first = _plan(ConvertOptions(model="org/m")).job_id
    assert _plan(ConvertOptions(model="org/m")).job_id == first
    assert _plan(ConvertOptions(model="org/m", num_splits=2)).job_id != first
    assert _plan(ConvertOptions(model="org/m"), model=_model(repo="Qwen/Qwen2.5-0.5B")).job_id != first


def test_family_model_and_licence_are_flagged() -> None:
    model = _model(repo="HuggingFaceTB/SmolLM2-360M-Instruct", model_type="llama", arch="LlamaForCausalLM",
                   weight_bytes=724_000_000, license_id="llama3.2")
    result = _plan(ConvertOptions(model="org/x"), model=model)
    assert result.recipe.kind == "family"
    assert any("family recipe" in w for w in result.warnings)
    assert any("licence 'llama3.2'" in w for w in result.warnings)


def test_other_machines_build_unverified_undeclared_bundles() -> None:
    x64 = HostInfo(system="Windows", native_arch="x64", python_platform="win-amd64", processor_name="Intel(R) Core",
                   total_ram_gb=64.0, available_ram_gb=40.0)
    with pytest.raises(NpuModelError) as exc:
        _plan(ConvertOptions(model="org/x"), host=x64)
    assert exc.value.reason_code == "NO_MATCHING_TARGET"
    result = _plan(ConvertOptions(model="org/x", target="qnn-x-elite"), host=x64)
    assert result.verify_level == "none"
    assert any("cannot be verified here" in w for w in result.warnings)


def test_memory_and_device_checks() -> None:
    big = _model(weight_bytes=16_000_000_000)  # ~8B params
    with pytest.raises(NpuModelError) as exc:
        _plan(ConvertOptions(model="org/x"), model=big)
    assert exc.value.reason_code == "NOT_ENOUGH_MEMORY"
    with pytest.raises(NpuModelError) as exc:
        _plan(ConvertOptions(model="org/x", device="cuda"))
    assert exc.value.reason_code == "NO_CUDA"
    with pytest.raises(NpuModelError) as exc:
        _plan(ConvertOptions(model="org/x", quality="recipe"))
    assert exc.value.reason_code == "QUALITY_NEEDS_CUDA"


def test_low_commit_headroom_is_flagged() -> None:
    from dataclasses import replace

    tight = replace(X_ELITE, available_commit_gb=2.0)
    result = _plan(ConvertOptions(model="org/x"), host=tight)
    assert any("Windows can commit only 2.0 GB" in w for w in result.warnings)
    roomy = replace(X_ELITE, available_commit_gb=20.0)
    assert not any("commit" in w for w in _plan(ConvertOptions(model="org/x"), host=roomy).warnings)


def test_estimates_match_m0_scale() -> None:
    est = estimates(_model())
    assert est["params_billions"] == 0.49 and 4.0 <= est["ram_gb"] <= 5.5 and 12 <= est["minutes"] <= 18


def test_job_receipt_records_and_resumes(tmp_path: Path) -> None:
    job = Job(tmp_path / "job", {"model": "m"})
    job.record("olive", "done", result={"seconds": 1.0})
    job.release()
    again = Job(tmp_path / "job", {"model": "m"})
    assert again.done("olive") and not again.done("assemble")
    assert again.data["stages"]["olive"]["result"]["seconds"] == 1.0


def test_job_lock_blocks_a_live_owner_and_takes_over_a_dead_one(tmp_path: Path) -> None:
    import subprocess
    import sys

    directory = tmp_path / "job"
    directory.mkdir()
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (directory / "job.lock").write_text(str(owner.pid), encoding="utf-8")
        with pytest.raises(NpuModelError) as exc:
            Job(directory, {})
        assert exc.value.reason_code == "JOB_RUNNING"
    finally:
        owner.kill()
        owner.wait()
    job = Job(directory, {})  # the owner died: its lock is stale
    assert (directory / "job.lock").read_text(encoding="utf-8") != str(owner.pid)
    job.release()
    assert not (directory / "job.lock").exists()
