from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest

from npu_model import gguf
from npu_model.core.errors import NpuModelError
from npu_model.host import HostInfo, resolve_target
from npu_model.profiles import get_target
from npu_model.sources import ModelRef, ResolvedModel
from npu_model.toolchains.llama_cpp import ConverterSource

X_ELITE = HostInfo(system="Windows", native_arch="arm64", python_platform="win-amd64",
                   processor_name="Snapdragon(R) X Elite - X1E78100 - Qualcomm(R) Oryon(TM) CPU",
                   total_ram_gb=15.4, available_ram_gb=6.0)


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPU_MODEL_HOME", str(tmp_path / "home"))


def _model(weight_bytes: int = 3_087_467_144) -> ResolvedModel:
    return ResolvedModel(ModelRef("hf", repo="Qwen/Qwen2.5-1.5B-Instruct"), commit="989aa7980e4c", gated=False,
                         license="apache-2.0", weight_bytes=weight_bytes,
                         config={"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
                                 "torch_dtype": "bfloat16", "max_position_embeddings": 32768})


def _payload(root: Path, complete: bool = True) -> Path:
    lib = root / "lib" / "ollama"
    (lib / "hexagon").mkdir(parents=True)
    exe = root / "ollama.exe"
    exe.write_bytes(b"")
    for name in ("llama-quantize.exe", "llama-server.exe", "llama-quantize", "llama-server"):
        (lib / name).write_bytes(b"")
    (lib / "hexagon" / "ggml-hexagon.dll").write_bytes(b"")
    for name in gguf.HTP_LIBRARIES[: None if complete else 2]:
        (lib / "hexagon" / name).write_bytes(b"")
    return exe


def _version(stdout: str) -> Any:
    def run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout, "")
    return run


def test_auto_target_stays_qnn_and_gguf_is_opt_in() -> None:
    assert resolve_target(X_ELITE).id == "qnn-x-elite"
    target = get_target("gguf-hexagon")
    assert target.format == "gguf" and target.matches(X_ELITE.processor_name)


def test_plan_needs_a_complete_runtime(tmp_path: Path) -> None:
    target = get_target("gguf-hexagon")
    with pytest.raises(NpuModelError) as exc:
        gguf.plan(_model(), X_ELITE, target, ollama_exe=None)
    assert exc.value.reason_code == "NEEDS_RUNTIME"
    partial = _payload(tmp_path / "partial", complete=False)
    with pytest.raises(NpuModelError) as exc:
        gguf.plan(_model(), X_ELITE, target, ollama_exe=partial)
    assert exc.value.reason_code == "RUNTIME_INCOMPLETE" and "libggml-htp-v79.so" in str(exc.value)


def test_plan_checks_the_runtime_commit_and_defaults(tmp_path: Path) -> None:
    exe = _payload(tmp_path / "fork")
    target = get_target("gguf-hexagon")
    plan = gguf.plan(_model(), X_ELITE, target, ollama_exe=exe,
                     run=_version("version: 0.6.0-dev (build 1, commit e467e5b)\nbuilt with Clang"))
    assert (plan.quant, plan.context, plan.verify_level) == ("Q4_0", 4096, "standard")
    assert any("first use: sets up the quantize toolchain" in w for w in plan.warnings)  # fresh home
    assert plan.destination.name == "qwen2.5-1.5b-instruct-gguf-hexagon"
    assert not any("pins" in w for w in plan.warnings)
    other = gguf.plan(_model(), X_ELITE, target, ollama_exe=exe, run=_version("version: 11521 (b42b7e6)"))
    assert any("b42b7e6" in w and "pins e467e5b5e" in w for w in other.warnings)
    assert other.job_id == plan.job_id and gguf.plan(_model(), X_ELITE, target, ollama_exe=exe, quant="Q8_0",
                                                     run=_version("")).job_id != plan.job_id
    with pytest.raises(NpuModelError) as exc:
        gguf.plan(_model(), X_ELITE, get_target("qnn-x-elite"), ollama_exe=exe)
    assert exc.value.reason_code == "WRONG_TARGET_FORMAT"


def test_estimates_scale_with_the_quant_type() -> None:
    q4, q8 = gguf.estimates(_model(), "Q4_0"), gguf.estimates(_model(), "Q8_0")
    assert q4["params_billions"] == 1.54 and 0.7 <= q4["gguf_gb"] <= 0.9 < q8["gguf_gb"]


def test_supported_architectures_reads_the_converter_registrations(tmp_path: Path) -> None:
    (tmp_path / "conversion").mkdir()
    (tmp_path / "conversion" / "qwen.py").write_text(
        '@ModelBase.register("Qwen2ForCausalLM", "Qwen2Model")\nclass Qwen2Model: ...\n'
        "@ModelBase.register('Qwen3ForCausalLM')\nclass Qwen3Model: ...\n", encoding="utf-8")
    (tmp_path / "convert_hf_to_gguf.py").write_text("print('hi')\n", encoding="utf-8")
    assert gguf.supported_architectures(tmp_path) == {"Qwen2ForCausalLM", "Qwen2Model", "Qwen3ForCausalLM"}


def test_assemble_writes_model_modelfile_licence_and_manifest(tmp_path: Path) -> None:
    exe = _payload(tmp_path / "fork")
    plan = gguf.plan(_model(), X_ELITE, get_target("gguf-hexagon"), ollama_exe=exe, run=_version(""))
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "LICENSE").write_text("apache", encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    quantized = tmp_path / "model-Q4_0.gguf"
    quantized.write_bytes(b"GGUF" + b"\0" * 64)
    bundle = gguf.assemble(quantized, snapshot, plan, {"converter_source": "e467e5b"})
    assert sorted(p.name for p in bundle.iterdir()) == ["LICENSE", "Modelfile", "model.gguf", gguf.MANIFEST]
    assert (bundle / "Modelfile").read_text(encoding="utf-8") == "FROM ./model.gguf\nPARAMETER num_ctx 4096\n"
    manifest = json.loads((bundle / gguf.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["format"] == "gguf" and manifest["precision"] == "Q4_0"
    assert manifest["runtime"]["llama_cpp"].startswith("e467e5b") and "model.gguf" in manifest["files"]


def test_converter_source_is_fetched_and_checked_file_by_file(tmp_path: Path) -> None:
    source = ConverterSource.load("b11644")
    assert source.commit.startswith("e467e5b") and "convert_hf_to_gguf.py" in source.files
    assert len(source.files) > 100 and source.problems()  # nothing fetched yet

    small = ConverterSource("t", "abc", "https://example.invalid/a.tar.gz",
                            {"convert_hf_to_gguf.py": __import__("hashlib").sha256(b"print(1)\n").hexdigest()})

    def archive(content: bytes) -> Any:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for name, data in (("llama.cpp-abc/convert_hf_to_gguf.py", content), ("llama.cpp-abc/other.txt", b"x")):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        buffer.seek(0)
        return lambda _url: buffer

    with pytest.raises(NpuModelError) as exc:
        small.ensure(opener=archive(b"print(2)\n"))
    assert exc.value.reason_code == "LLAMA_CPP_MISMATCH"
    assert small.ensure(opener=archive(b"print(1)\n")) == small.directory
    assert not small.problems() and not (small.directory / "other.txt").exists()
    assert small.environment()["PYTHONPATH"].endswith("gguf-py")
