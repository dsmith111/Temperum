from __future__ import annotations

import json
import stat
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from npu_model.cli.main import app
from npu_model.core.errors import NpuModelError
from npu_model.inspect import prebuilt


class Proto(SimpleNamespace):
    def HasField(self, name: str) -> bool:
        return hasattr(self, name)


def context_node(**updates: object) -> Proto:
    values = {
        "source": Proto(name="source", type=3, s=b"QNN"),
        "embed_mode": Proto(name="embed_mode", type=2, i=0),
        "main_context": Proto(name="main_context", type=2, i=1),
        "ep_cache_context": Proto(name="ep_cache_context", type=3, s=b"context.qnn"),
    }
    values.update(updates)  # type: ignore[arg-type]
    for value in values.values():
        value.tensors = []
        value.graphs = []
        value.sparse_tensors = []
    return Proto(op_type="EPContext", domain="com.microsoft", attribute=list(values.values()))


def graph_model(node: Proto | None = None) -> Proto:
    return Proto(ir_version=10, functions=[], training_info=[],
                 graph=Proto(node=[node or context_node()], initializer=[], sparse_initializer=[]))


@pytest.fixture
def fake_onnx(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Policy/API seam only: this stub is never native/valid-artifact evidence."""
    module = SimpleNamespace(
        load=MagicMock(return_value=graph_model()), TensorProto=SimpleNamespace(EXTERNAL=1),
        AttributeProto=SimpleNamespace(STRING=3, INT=2),
        checker=SimpleNamespace(check_model=MagicMock(), ValidationError=ValueError),
    )
    monkeypatch.setitem(sys.modules, "onnx", module)
    monkeypatch.setitem(sys.modules, "google.protobuf.message", SimpleNamespace(DecodeError=ValueError))
    return module


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "prebuilt"
    root.mkdir()
    (root / "model_ctx.onnx").write_bytes(b"stub-graph-API-policy-test-only")
    (root / "context.qnn").write_bytes(b"stub-context-never-native-certified")
    (root / "tokenizer.json").write_text('{"version":"1.0","model":{"type":"BPE","vocab":{}}}')
    config = {"model": {"type": "llama", "context_length": 4096, "vocab_size": 10,
                        "eos_token_id": 2, "decoder": {"filename": "model_ctx.onnx",
                            "session_options": {"provider_options": [{"qnn": {"backend_type": "htp"}}]}}},
              "search": {"max_length": 4096}}
    (root / "genai_config.json").write_text(json.dumps(config))
    profile = {"schema_version": 1, "platform": "windows/arm64", "provider": "qnn",
               "genai_version": "0.12.2", "ort_version": "1.24.1",
               "processor_names": ["Exact Engineering CPU"], "context_length": 4096,
               "precision": "publisher-declared-mixed", "capabilities": ["completion"],
               "stages": [{"filename": "model_ctx.onnx", "provider": "qnn"}],
               "source": {"repository": "https://example.invalid/repo", "revision": "a" * 40}}
    (root / "ollama_npu.json").write_text(json.dumps(profile))
    return root


def change(path: Path, *fields: str, value: object) -> None:
    document = json.loads(path.read_text())
    cursor = document
    for field in fields[:-1]:
        cursor = cursor[field]
    cursor[fields[-1]] = value
    path.write_text(json.dumps(document))


def codes(report: prebuilt.Inspection) -> set[str]:
    return {error["code"] for error in report.errors}


def test_missing_dependency_fails_with_inventory_and_declaration(
    bundle: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "onnx", None)
    report = prebuilt.inspect_prebuilt(bundle)
    assert not report.importable and report.structural == "failed"
    assert codes(report) == {"MISSING_ONNX"}
    assert report.declaration_status == "consistent"
    assert report.content_fingerprint and len(report.assets) == 5
    assert report.to_dict()["levels"]["native_evidence"]["certified"] is False


def test_inspect_json_is_readonly_and_truthful(bundle: Path, fake_onnx: SimpleNamespace) -> None:
    before = prebuilt.inventory(bundle)
    result = CliRunner().invoke(app, ["inspect-prebuilt", "--input", str(bundle)])
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["levels"]["structural"]["status"] == "passed"
    assert value["levels"]["declaration"]["status"] == "consistent"
    assert value["levels"]["target"]["status"] == "runtime-match-processor-unchecked"
    assert value["levels"]["native_evidence"]["status"] == "not-run"
    assert prebuilt.inventory(bundle) == before
    fake_onnx.load.assert_called_once_with(str(bundle / "model_ctx.onnx"), load_external_data=False)
    fake_onnx.checker.check_model.assert_called_once_with(str(bundle / "model_ctx.onnx"), full_check=False)


@pytest.mark.parametrize("name,code", [
    ("genai_config.json", "MISSING_ASSET"), ("tokenizer.json", "MISSING_TOKENIZER"),
    ("context.qnn", "MISSING_ASSET"), ("model_ctx.onnx", "MISSING_ASSET"),
])
def test_missing_required_assets(bundle: Path, fake_onnx: SimpleNamespace, name: str, code: str) -> None:
    (bundle / name).unlink()
    assert code in codes(prebuilt.inspect_prebuilt(bundle))


@pytest.mark.parametrize("name", ["model_ctx.onnx", "genai_config.json", "tokenizer.json", "context.qnn"])
def test_zero_required_assets(bundle: Path, fake_onnx: SimpleNamespace, name: str) -> None:
    (bundle / name).write_bytes(b"")
    assert not prebuilt.inspect_prebuilt(bundle).importable


@pytest.mark.parametrize("reference", ["../escape.onnx", "/abs.onnx", "C:/bad.onnx",
                                      r"..\bad.onnx", "folder//a.onnx", "folder/./a.onnx",
                                      "CON.onnx", "bad:.onnx", "bad .onnx."])
def test_unsafe_config_references(bundle: Path, fake_onnx: SimpleNamespace, reference: str) -> None:
    change(bundle / "genai_config.json", "model", "decoder", "filename", value=reference)
    assert "UNSAFE_REFERENCE" in codes(prebuilt.inspect_prebuilt(bundle))


@pytest.mark.parametrize("content", [
    '{"model":{},"model":{}}', '{"model":{"x":NaN}}', "[]", '{"unfinished":',
    "[" * 40 + "]" * 40,
])
def test_invalid_bounded_json(bundle: Path, content: str) -> None:
    (bundle / "genai_config.json").write_text(content)
    assert not prebuilt.inspect_prebuilt(bundle).importable


@pytest.mark.parametrize("failure", ["decode", "checker", "empty", "missing-context", "raw-qnn"])
def test_invalid_graph_failures(bundle: Path, fake_onnx: SimpleNamespace, failure: str) -> None:
    if failure == "decode":
        fake_onnx.load.side_effect = ValueError("invalid protobuf")
    elif failure == "checker":
        fake_onnx.checker.check_model.side_effect = ValueError("invalid graph shape")
    elif failure == "empty":
        fake_onnx.load.return_value.graph.node = []
    elif failure == "missing-context":
        fake_onnx.load.return_value = graph_model(context_node(
            ep_cache_context=Proto(name="ep_cache_context", type=3, s=b"")))
    else:
        fake_onnx.load.return_value = graph_model(Proto(op_type="Add", domain="", attribute=[]))
    report = prebuilt.inspect_prebuilt(bundle)
    assert not report.importable
    assert codes(report) & {"INVALID_GRAPH", "MISSING_ASSET", "QNN_CONTEXT_REQUIRED"}


@pytest.mark.parametrize("reference", ["../outside.bin", r"..\outside.bin", "/abs.bin", "absent.bin"])
def test_external_context_paths_fail(bundle: Path, fake_onnx: SimpleNamespace, reference: str) -> None:
    fake_onnx.load.return_value = graph_model(context_node(ep_cache_context=Proto(
        name="ep_cache_context", type=3, s=reference.encode())))
    assert not prebuilt.inspect_prebuilt(bundle).importable


@pytest.mark.parametrize("metadata,valid", [
    ({"location": "weights.dat", "offset": "1", "length": "3"}, True),
    ({"location": "../weights.dat"}, False), ({"location": "missing.dat"}, False),
    ({"location": "weights.dat", "offset": "8", "length": "1"}, False),
    ({"location": "weights.dat", "length": "-1"}, False),
])
def test_cpu_external_tensor_references(
    bundle: Path, fake_onnx: SimpleNamespace, metadata: dict[str, str], valid: bool,
) -> None:
    (bundle / "ollama_npu.json").unlink()
    change(bundle / "genai_config.json", "model", "decoder", "session_options", value={})
    (bundle / "weights.dat").write_bytes(b"12345")
    model = graph_model(Proto(op_type="Identity", domain="", attribute=[]))
    model.graph.initializer = [Proto(data_location=1, external_data=[
        Proto(key=key, value=value) for key, value in metadata.items()])]
    fake_onnx.load.return_value = model
    report = prebuilt.inspect_prebuilt(bundle)
    assert report.importable is valid
    if valid:
        assert report.graphs[0]["external_assets"] == ["weights.dat"]


@pytest.mark.parametrize("checker_accepts", [True, False])
def test_embedded_tensor_ignores_inactive_external_metadata(
    bundle: Path, fake_onnx: SimpleNamespace, checker_accepts: bool,
) -> None:
    fake_onnx.load.return_value.graph.initializer = [Proto(
        data_location=0, raw_data=b"embedded",
        external_data=[Proto(key="location", value="absent-old-weights.dat")])]
    if not checker_accepts:
        fake_onnx.checker.check_model.side_effect = ValueError("invalid embedded data")
    report = prebuilt.inspect_prebuilt(bundle)
    assert report.importable is checker_accepts
    fake_onnx.checker.check_model.assert_called_once()
    if checker_accepts:
        assert report.graphs[0]["external_assets"] == ["context.qnn"]
    else:
        assert "INVALID_GRAPH" in codes(report)


def test_duplicate_external_metadata_rejected(bundle: Path, fake_onnx: SimpleNamespace) -> None:
    fake_onnx.load.return_value.graph.initializer = [Proto(data_location=1, external_data=[
        Proto(key="location", value="context.qnn"), Proto(key="location", value="other")])]
    assert "INVALID_GRAPH" in codes(prebuilt.inspect_prebuilt(bundle))


def test_nested_hybrid_pipeline_is_preserved(bundle: Path, fake_onnx: SimpleNamespace) -> None:
    (bundle / "cpu").mkdir()
    (bundle / "cpu" / "embedding.onnx").write_bytes(b"stubCPU")
    (bundle / "cpu" / "head.onnx").write_bytes(b"stubCPU")
    change(bundle / "genai_config.json", "model", "type", value="decoder-pipeline")
    change(bundle / "genai_config.json", "model", "decoder", value={"pipeline": [
        {"embedding": {"filename": "cpu/embedding.onnx", "session_options": {}}},
        {"context": {"filename": "model_ctx.onnx",
                     "session_options": {"provider_options": [{"qnn": {"backend_path": "QnnHtp.dll"}}]}}},
        {"head": {"filename": "cpu/head.onnx", "session_options": {}}},
    ]})
    change(bundle / "ollama_npu.json", "stages", value=[
        {"filename": "cpu/embedding.onnx", "provider": "cpu"},
        {"filename": "model_ctx.onnx", "provider": "qnn"}, {"filename": "cpu/head.onnx", "provider": "cpu"}])
    fake_onnx.load.side_effect = lambda path, **kwargs: graph_model(
        context_node() if path.endswith("model_ctx.onnx") else Proto(op_type="Identity", domain="", attribute=[]))
    before = prebuilt.inventory(bundle)
    report = prebuilt.inspect_prebuilt(bundle, processor_name="Exact Engineering CPU")
    assert report.importable, report.errors
    assert len(report.graphs) == 3 and report.target["status"] == "declared-match"
    assert report.declaration["source"]["revision"] == "a" * 40
    assert prebuilt.inventory(bundle) == before


@pytest.mark.parametrize("field,value,code", [
    ("context_length", 2048, "CONTRADICTORY_DECLARATION"),
    ("schema_version", True, "INVALID_DECLARATION"),
    ("producer", "guessed-producer", "INVALID_DECLARATION"),
    ("precision", "", "INVALID_DECLARATION"),
    ("processor_names", ["CPU", "cpu"], "INVALID_DECLARATION"),
    ("capabilities", ["embedding"], "INVALID_DECLARATION"),
    ("stages", [{"filename": "model_ctx.onnx", "provider": "cpu"}], "CONTRADICTORY_DECLARATION"),
    ("source", {"repository": "https://user:secret@example.invalid/repo", "revision": "a" * 40},
     "INVALID_DECLARATION"),
])
def test_declaration_consistency(
    bundle: Path, fake_onnx: SimpleNamespace, field: str, value: object, code: str,
) -> None:
    change(bundle / "ollama_npu.json", field, value=value)
    report = prebuilt.inspect_prebuilt(bundle)
    assert report.structural == "passed" and not report.importable
    assert code in codes(report)


@pytest.mark.parametrize("field,value", [("ort_version", "1.24.0"), ("genai_version", "0.12.1")])
def test_target_runtime_mismatch(bundle: Path, fake_onnx: SimpleNamespace, field: str, value: str) -> None:
    change(bundle / "ollama_npu.json", field, value=value)
    report = prebuilt.inspect_prebuilt(bundle)
    assert report.target["status"] == "mismatch" and not report.importable


def test_target_processor_mismatch_and_absent_profile(bundle: Path, fake_onnx: SimpleNamespace) -> None:
    assert not prebuilt.inspect_prebuilt(bundle, processor_name="Other CPU").importable
    (bundle / "ollama_npu.json").unlink()
    report = prebuilt.inspect_prebuilt(bundle, processor_name="Other CPU")
    assert report.importable and report.target["status"] == "unverified"


def test_full_identity_changes_for_supplemental_tamper(bundle: Path) -> None:
    first = prebuilt.inventory(bundle)
    with pytest.raises(FrozenInstanceError):
        first[0].sha256 = "changed"  # type: ignore[misc]
    (bundle / "LICENSE").write_text("license text")
    second = prebuilt.inventory(bundle)
    assert prebuilt.content_fingerprint(first) != prebuilt.content_fingerprint(second)
    (bundle / "context.qnn").write_bytes(b"tampered-context")
    assert prebuilt.content_fingerprint(second) != prebuilt.content_fingerprint(prebuilt.inventory(bundle))


def test_reparse_point_rejected(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.lstat
    def lstat(path: Path) -> object:
        if path == bundle / "context.qnn":
            return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0x400)
        return original(path)
    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(NpuModelError, match="Symlink"):
        prebuilt.regular_path(bundle / "context.qnn")


def test_entry_count_bound(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prebuilt, "MAX_FILES", 2)
    assert "INSPECTION_LIMIT" in codes(prebuilt.inspect_prebuilt(bundle))

def test_inventory_reparse_and_hash_mutation_fail(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.lstat
    def lstat(path: Path) -> object:
        if path == bundle / "context.qnn":
            return SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=0x400)
        return original(path)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "lstat", lstat)
        assert "SYMLINK_ASSET" in codes(prebuilt.inspect_prebuilt(bundle))
    original_hash = prebuilt.sha256_file
    def mutate(path: Path) -> str:
        digest = original_hash(path)
        path.write_bytes(path.read_bytes() + b"changed during hashing")
        return digest
    monkeypatch.setattr(prebuilt, "sha256_file", mutate)
    assert "ASSET_CHANGED" in codes(prebuilt.inspect_prebuilt(bundle))


@pytest.mark.parametrize("name", ["model.gguf", "model.safetensors", "native.dll", "native.exe"])
def test_mixed_formats_fail(bundle: Path, fake_onnx: SimpleNamespace, name: str) -> None:
    (bundle / name).write_bytes(b"mixed")
    assert "INVALID_ASSET" in codes(prebuilt.inspect_prebuilt(bundle))


@pytest.mark.parametrize("key,value", [("backend_type", "cpu"), ("backend_path", "other.dll")])
def test_declared_htp_backend_contradictions(
    bundle: Path, fake_onnx: SimpleNamespace, key: str, value: str,
) -> None:
    change(bundle / "genai_config.json", "model", "decoder", "session_options",
           value={"provider_options": [{"qnn": {key: value}}]})
    assert "CONTRADICTORY_DECLARATION" in codes(prebuilt.inspect_prebuilt(bundle))


def test_duplicate_graph_references_and_providers(bundle: Path, fake_onnx: SimpleNamespace) -> None:
    change(bundle / "genai_config.json", "model", "decoder", "session_options",
           value={"provider_options": [{"qnn": {}}, {"QNN": {}}]})
    assert "INVALID_CONFIG" in codes(prebuilt.inspect_prebuilt(bundle))
    change(bundle / "genai_config.json", "model", "type", value="decoder-pipeline")
    change(bundle / "genai_config.json", "model", "decoder", value={"pipeline": [
        {"a": {"filename": "model_ctx.onnx"}}, {"b": {"filename": "model_ctx.onnx"}}]})
    assert "DUPLICATE_STAGE" in codes(prebuilt.inspect_prebuilt(bundle))


def test_graph_size_bound(bundle: Path, fake_onnx: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prebuilt, "MAX_GRAPH", 2)
    assert "INSPECTION_LIMIT" in codes(prebuilt.inspect_prebuilt(bundle))
    fake_onnx.load.assert_not_called()


def test_duplicate_pipeline_and_unreferenced_graph(bundle: Path, fake_onnx: SimpleNamespace) -> None:
    (bundle / "unused.onnx").write_bytes(b"extra")
    assert "UNREFERENCED_GRAPH" in codes(prebuilt.inspect_prebuilt(bundle))
    (bundle / "unused.onnx").unlink()
    change(bundle / "genai_config.json", "model", "type", value="decoder-pipeline")
    change(bundle / "genai_config.json", "model", "decoder", value={"pipeline": [
        {"a": {"filename": "model_ctx.onnx"}}, {"a": {"filename": "model_ctx.onnx"}}]})
    assert "DUPLICATE_STAGE" in codes(prebuilt.inspect_prebuilt(bundle))
