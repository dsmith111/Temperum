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
from npu_model.gguf_verify import runtime_env
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


def test_auto_target_follows_the_forks_hexagon_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_target(X_ELITE).id == "qnn-x-elite"
    monkeypatch.setenv("OLLAMA_HEXAGON", "1")
    assert resolve_target(X_ELITE).id == "gguf-hexagon"
    assert resolve_target(X_ELITE, fmt="qnn").id == "qnn-x-elite"  # the QNN pipeline resolves its own auto


def test_plan_needs_a_complete_runtime(tmp_path: Path) -> None:
    target = get_target("gguf-hexagon")
    with pytest.raises(NpuModelError) as exc:
        gguf.plan(_model(), X_ELITE, target, ollama_exe=None)
    assert exc.value.reason_code == "NEEDS_RUNTIME"
    partial = _payload(tmp_path / "partial", complete=False)
    with pytest.raises(NpuModelError) as exc:
        gguf.plan(_model(), X_ELITE, target, ollama_exe=partial, run=_version(""))
    assert exc.value.reason_code == "RUNTIME_INCOMPLETE" and "npu setup --provider hexagon" in str(exc.value)


def _setup_report(status: str, directory: Path | None = None, stage: str = "htp_integrity") -> Any:
    """What `ollama npu setup --provider hexagon --validate-only --json` prints."""
    return _version(json.dumps({"status": "files_verified" if status == "pass" else "checks_failed",
                                "dir": str(directory or ""), "checks": [{"stage": stage, "status": status}]}))


def test_the_signed_htp_set_is_where_the_fork_says(tmp_path: Path) -> None:
    exe = _payload(tmp_path / "fork", complete=False)  # a package: the backend without the signed set
    backend = exe.parent / "lib" / "ollama" / "hexagon"
    installed = tmp_path / "htp"
    assert gguf.signed_htp_set(exe, backend, _setup_report("pass", installed)) == installed
    assert gguf.signed_htp_set(exe, backend, _setup_report("fail")) is None
    # Older builds answer --validate-only for the QNN runtime; only a bundled set counts for them.
    qnn = _setup_report("pass", tmp_path / "qnn", stage="runtime_integrity")
    assert gguf.signed_htp_set(exe, backend, qnn) is None
    bundled = _payload(tmp_path / "bundled")
    bundled_backend = bundled.parent / "lib" / "ollama" / "hexagon"
    assert gguf.signed_htp_set(bundled, bundled_backend, qnn) == bundled_backend

    payload = gguf.LlamaPayload.from_executable(exe, _setup_report("pass", installed))
    assert payload.htp == installed and payload.problems() == []
    assert runtime_env(payload)["ADSP_LIBRARY_PATH"].startswith(str(installed))


def test_plan_checks_the_runtime_commit_and_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = _payload(tmp_path / "fork")
    target = get_target("gguf-hexagon")
    plan = gguf.plan(_model(), X_ELITE, target, ollama_exe=exe,
                     run=_version("version: 0.6.0-dev (build 1, commit e467e5b)\nbuilt with Clang"))
    assert (plan.quant, plan.context, plan.verify_level) == ("Q4_0", 4096, "standard")
    assert any("first use: sets up the quantize toolchain" in w for w in plan.warnings)  # fresh home
    assert any("OLLAMA_HEXAGON=1" in w for w in plan.warnings)  # the fork keeps its Hexagon backend off
    assert plan.destination.name == "qwen2.5-1.5b-instruct-gguf-hexagon"
    assert not any("pins" in w or "not been verified" in w for w in plan.warnings)
    monkeypatch.setenv("OLLAMA_HEXAGON", "true")
    assert not any("OLLAMA_HEXAGON" in w for w in gguf.plan(_model(), X_ELITE, target, ollama_exe=exe,
                                                             run=_version("commit e467e5b")).warnings)
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


def test_quick_verification_generates_with_the_bundles_own_chat_format(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from npu_model import gguf_verify

    sent: list[Any] = []

    class FakeServer:
        def __init__(self, *_args: Any) -> None: ...
        def __enter__(self) -> FakeServer: return self
        def __exit__(self, *_exc: object) -> None: ...
        def text(self) -> str:
            return ("load_tensors: offloaded 29/29 layers to GPU\n"
                    "load_tensors:         HTP0 model buffer size =   898.14 MiB\n")
        def apply_template(self, messages: list[dict[str, str]]) -> str:
            return f"<user>{messages[-1]['content']}<assistant>"
        def complete(self, prompt: Any, n_predict: int) -> dict[str, Any]:
            sent.append(prompt)
            return {"content": "Paris", "tokens": [1, 2, 3], "timings": {"predicted_per_second": 30.0}}

    monkeypatch.setattr(gguf_verify, "LlamaServer", FakeServer)
    plan = SimpleNamespace(verify_level="quick", payload=None, context=4096)
    report = gguf_verify.verify_gguf(tmp_path / "bundle", tmp_path / "source", plan, None,  # type: ignore[arg-type]
                                     tmp_path / "work", lambda _message: None)
    # No source model runs at quick level, so the prompts go through the GGUF's own chat template.
    assert report["verdict"] == "passed", report["reasons"]
    assert len(sent) == 3 and all(p.startswith("<user>") for p in sent)
    assert "greedy_exact" not in report["quality"] and report["speed"]["median_tokens_per_s"] == 30.0


def test_a_model_too_large_to_run_here_is_still_checked_with_its_tokenizer(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from npu_model import gguf_verify
    from npu_model.verify import CONVERSATIONS

    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(json.dumps({"text": "Some held-out text."}) + "\n", encoding="utf-8")
    monkeypatch.setattr(gguf_verify.calib, "ensure_corpus", lambda *_a: SimpleNamespace(path=heldout))
    scripts: list[list[str]] = []

    def run_script(_python: Path, script: str, args: list[str], *_rest: Any) -> tuple[int, str]:
        scripts.append([script, *args])
        out = Path(args[args.index("--out") + 1])
        out.write_text(json.dumps({  # what reference.py --tokenizer-only writes: inputs, no answers
            "tokenizer_only": True, "prompts": [{"prompt": "Q", "input_ids": [7, 8, 9]}],
            "needle": [{"target": 900, "input_len": 870, "input_ids": [5] * 870},
                       {"target": 1800, "input_len": 1760, "input_ids": [5] * 1760}],
            "rendered": [f"<user>{c[-1]['content']}<assistant>" for c in CONVERSATIONS]}), encoding="utf-8")
        return 0, ""

    sent: list[Any] = []

    class FakeServer:
        def __init__(self, *_args: Any) -> None: ...
        def __enter__(self) -> FakeServer: return self
        def __exit__(self, *_exc: object) -> None: ...
        def text(self) -> str:
            return ("load_tensors: offloaded 33/33 layers to GPU\n"
                    "load_tensors:         HTP0 model buffer size =  2100.00 MiB\n")
        def apply_template(self, messages: list[dict[str, str]]) -> str:
            return f"<user>{messages[-1]['content']}<assistant>"
        def complete(self, prompt: Any, n_predict: int) -> dict[str, Any]:
            sent.append(prompt)
            long = len(prompt) > 100
            return {"content": "4721" if long else "Paris", "tokens": [1, 2, 3],
                    "timings": {"predicted_per_second": 20.0, "prompt_ms": 1760.0 if long else 50.0}}

    monkeypatch.setattr(gguf_verify, "run_script", run_script)
    monkeypatch.setattr(gguf_verify, "LlamaServer", FakeServer)
    monkeypatch.setattr(gguf_verify, "quantize", lambda *_a, **_k: None)
    monkeypatch.setattr(gguf_verify, "perplexity", lambda *_a, npu, **_k: 12.6 if npu else 10.0)
    plan = SimpleNamespace(verify_level="standard", payload=None, context=4096, quant="Q4_0",
                           estimates={"params_billions": 3.8},
                           host=SimpleNamespace(available_ram_gb=3.0, total_ram_gb=16.0))
    chains = SimpleNamespace(python_exe=lambda _role: Path("python"))
    report = gguf_verify.verify_gguf(tmp_path / "bundle", tmp_path / "source", plan,  # type: ignore[arg-type]
                                     chains, tmp_path / "work", lambda _message: None,
                                     source_gguf=tmp_path / "model-16bit.gguf")
    assert "--tokenizer-only" in scripts[0] and "--dtype" not in scripts[0]
    assert sent[0] == [7, 8, 9]  # the source tokenizer's inputs, not the server's own rendering
    assert "greedy_exact" not in report["quality"]
    assert [n["correct"] for n in report["quality"]["needle"]] == [True, True]
    assert report["template"]["status"] == "match"
    assert report["quality"]["perplexity"]["ratio"] == 1.26
    assert report["speed"]["long_prompt_tokens"] == 1760 and report["speed"]["long_prompt_tokens_per_s"] == 1000.0
    assert report["verdict"] == "warning"
    assert any(r.startswith("answers not compared with the source model") for r in report["reasons"])
    assert any("--quant Q4_K_M" in r for r in report["reasons"])
