from __future__ import annotations

import subprocess

import pytest

from npu_model.core.errors import NpuModelError
from npu_model.host import HostInfo, detect_host, resolve_target
from npu_model.profiles import DEFAULT_RUNTIME, RUNTIMES, TARGETS, get_runtime, get_target

X1E = "Snapdragon(R) X Elite - X1E78100 - Qualcomm(R) Oryon(TM) CPU"


def _host(name: str | None) -> HostInfo:
    return HostInfo(system="Windows", native_arch="arm64", python_platform="win-arm64", processor_name=name)


def test_x_elite_processor_resolves_to_v73_target() -> None:
    target = resolve_target(_host(X1E))
    assert target.id == "qnn-x-elite"
    assert (target.soc_model, target.htp_arch) == ("60", "v73")


def test_x_plus_shares_the_x_elite_profile() -> None:
    assert resolve_target(_host("Snapdragon(R) X Plus - X1P64100 - Qualcomm(R) Oryon(TM) CPU")).id == "qnn-x-elite"


def test_x2_elite_resolves_to_its_own_profile() -> None:
    target = resolve_target(_host("Snapdragon(R) X2 Elite - X2E88100 - Qualcomm(R) Oryon(TM) CPU"))
    assert target.id == "qnn-x2-elite" and target.soc_model == "87" and not target.verified


@pytest.mark.parametrize("name", [None, "Intel(R) Core(TM) Ultra 7 155H", "AMD Ryzen AI 9 HX 370"])
def test_unmatched_machine_requires_explicit_target(name: str | None) -> None:
    with pytest.raises(NpuModelError) as exc:
        resolve_target(_host(name))
    assert exc.value.reason_code == "NO_MATCHING_TARGET"
    assert "--target" in (exc.value.hint or "")


def test_explicit_target_overrides_detection() -> None:
    assert resolve_target(_host("Intel(R) Core(TM) Ultra 7 155H"), "qnn-x2-elite").id == "qnn-x2-elite"


def test_unknown_profile_ids_fail_with_known_list() -> None:
    with pytest.raises(NpuModelError) as exc:
        get_target("qnn-nope")
    assert "qnn-x-elite" in (exc.value.hint or "")
    with pytest.raises(NpuModelError):
        get_runtime("nope")


def test_default_runtime_compiles_with_the_fork_qnn_build() -> None:
    runtime = get_runtime(DEFAULT_RUNTIME)
    assert (runtime.genai_version, runtime.ort_version, runtime.qnn_version) == ("0.12.2", "1.24.1", "2.42.0")
    assert "onnxruntime-qnn==1.24.1" in runtime.compile_requirements
    assert "onnxruntime-qnn==1.24.1" in runtime.verify_requirements
    assert runtime.builder_genai == runtime.genai_version


def test_every_target_profile_is_well_formed() -> None:
    for target in TARGETS.values():
        assert target.processor_patterns and target.format in ("qnn", "gguf")
        if target.format == "qnn":
            assert target.provider == "qnn" and target.soc_model.isdigit()
    assert DEFAULT_RUNTIME in RUNTIMES


def test_detect_host_survives_failing_probes() -> None:
    def failing_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("no powershell")

    host = detect_host(run=failing_run)
    assert host.npu_devices == [] and host.cuda_gpus == []
    assert host.native_arch and host.python_platform
