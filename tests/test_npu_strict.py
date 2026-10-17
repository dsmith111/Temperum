from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from npu_model.core.errors import NpuModelError
from npu_model.validate.npu_strict import validate_npu_strict


class TestNpuStrictValidator:

    def _make_bundle(self, tmp_path: Path, *, with_onnx: bool = True,
                     with_ctx: bool = False, with_bins: bool = False,
                     with_data: bool = False, with_genai: bool = True) -> Path:
        d = tmp_path / "bundle"
        d.mkdir()
        if with_onnx:
            (d / "model.onnx").write_bytes(b"\x00" * 16)
        if with_ctx:
            (d / "model_ctx.onnx").write_bytes(b"\x00" * 16)
        if with_bins:
            (d / "model_qnn.bin").write_bytes(b"\x00" * 32)
        if with_data:
            (d / "model.onnx.data").write_bytes(b"\x00" * 100)
        if with_genai:
            cfg = {"model": {"decoder": {"session_options": {}}}, "search": {}}
            (d / "genai_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (d / "tokenizer.json").write_text("{}", encoding="utf-8")
        return d

    def test_no_onnx_files_fails(self, tmp_path: Path) -> None:
        d = self._make_bundle(tmp_path, with_onnx=False)
        result = validate_npu_strict(d)
        assert not result.passed

    def test_basic_onnx_present(self, tmp_path: Path) -> None:
        d = self._make_bundle(tmp_path)
        result = validate_npu_strict(d)
        # Should have some checks
        assert len(result.checks) > 0
        # Onnx present check should pass
        onnx_check = next(c for c in result.checks if c.name == "onnx_present")
        assert onnx_check.status == "OK"

    def test_warns_no_ctx_graphs(self, tmp_path: Path) -> None:
        d = self._make_bundle(tmp_path, with_bins=True)
        result = validate_npu_strict(d)
        ctx_check = next(c for c in result.checks if c.name == "context_cache")
        assert ctx_check.status == "FAIL"

    def test_ctx_graphs_ok(self, tmp_path: Path) -> None:
        d = self._make_bundle(tmp_path, with_ctx=True, with_bins=True)
        result = validate_npu_strict(d)
        ctx_check = next(c for c in result.checks if c.name == "context_cache")
        assert ctx_check.status == "OK"

    def test_warns_data_files_present(self, tmp_path: Path) -> None:
        d = self._make_bundle(tmp_path, with_data=True)
        result = validate_npu_strict(d)
        data_check = next(c for c in result.checks if c.name == "external_data")
        assert data_check.status == "FAIL"

    def test_no_data_files_ok(self, tmp_path: Path) -> None:
        d = self._make_bundle(tmp_path)
        result = validate_npu_strict(d)
        data_check = next(c for c in result.checks if c.name == "external_data")
        assert data_check.status == "OK"

    def test_result_properties(self, tmp_path: Path) -> None:
        d = self._make_bundle(tmp_path, with_data=True)
        result = validate_npu_strict(d)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)


@pytest.fixture()
def configured_bundle(tmp_path: Path) -> Path:
    """Filename/config fixtures only; no real ONNX or QNN binary is generated."""
    (tmp_path / "model_ctx.onnx").write_bytes(b"mock-context-graph")
    (tmp_path / "model_qnn.bin").write_bytes(b"mock-context-binary")
    (tmp_path / "genai_config.json").write_text(json.dumps({
        "model": {"decoder": {"session_options": {
            "provider_options": [{"qnn": {}}],
            "disable_cpu_ep_fallback": "1",
        }}},
    }), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("missing", ["onnx", "onnxruntime"])
def test_missing_mandatory_module_cannot_pass(
    configured_bundle: Path, missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npu_model.core import shapes

    if missing == "onnx":
        monkeypatch.setitem(sys.modules, "onnx", None)
    else:
        monkeypatch.setattr(shapes, "has_dynamic_shapes", lambda path: [])
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    result = validate_npu_strict(configured_bundle)
    assert not result.passed
    assert any(check.status == "SKIP" for check in result.checks)
    assert result.errors


@pytest.mark.parametrize("retained_providers", [["CPUExecutionProvider"], ["QNNExecutionProvider"]])
def test_qnn_session_checks_actual_provider_retention(
    configured_bundle: Path, retained_providers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock policy test only: passing here is not hardware execution evidence."""
    from npu_model.core import shapes

    monkeypatch.setattr(shapes, "has_dynamic_shapes", lambda path: [])
    ort = MagicMock()
    ort.get_available_providers.return_value = ["QNNExecutionProvider", "CPUExecutionProvider"]
    ort.InferenceSession.return_value.get_providers.return_value = retained_providers
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setenv("NPU_QNN_BACKEND_PATH", "custom-QnnHtp.dll")
    result = validate_npu_strict(configured_bundle)
    assert result.passed == ("QNNExecutionProvider" in retained_providers)
    ort.SessionOptions.return_value.add_session_config_entry.assert_called_once_with(
        "session.disable_cpu_ep_fallback", "1"
    )
    call_options = ort.InferenceSession.call_args.kwargs
    assert call_options["providers"] == ["QNNExecutionProvider"]
    assert call_options["provider_options"] == [{"backend_path": "custom-QnnHtp.dll"}]


def test_qnn_provider_unavailable_is_incomplete(
    configured_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npu_model.core import shapes

    monkeypatch.setattr(shapes, "has_dynamic_shapes", lambda path: [])
    ort = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    result = validate_npu_strict(configured_bundle)
    assert not result.passed
    assert next(check for check in result.checks if check.name == "qnn_session_load").status == "SKIP"


def test_ort_native_error_is_reported_as_failed_check(
    configured_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npu_model.core import shapes

    class EPFail(Exception):
        pass

    monkeypatch.setattr(shapes, "has_dynamic_shapes", lambda path: [])
    ort = MagicMock()
    ort.get_available_providers.return_value = ["QNNExecutionProvider"]
    ort.capi.onnxruntime_pybind11_state.EPFail = EPFail
    ort.InferenceSession.side_effect = EPFail("QNN backend initialization failed")
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    result = validate_npu_strict(configured_bundle)
    assert not result.passed
    assert next(check for check in result.checks if check.name.startswith("qnn_load_")).status == "FAIL"


def test_unreadable_graph_is_a_failed_check(
    configured_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npu_model.core import shapes

    def unreadable(path: Path) -> list[str]:
        raise NpuModelError("shapes", "ONNX_INSPECTION_FAILED", "not an ONNX graph")

    monkeypatch.setattr(shapes, "has_dynamic_shapes", unreadable)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    result = validate_npu_strict(configured_bundle)
    check = next(c for c in result.checks if c.name.startswith("shapes_"))
    assert check.status == "FAIL"
    assert not result.passed


@pytest.mark.parametrize("config", ["{", "[]", '{"model": null}'])
def test_malformed_config_is_not_advisory(
    configured_bundle: Path, config: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "onnx", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    (configured_bundle / "genai_config.json").write_text(config, encoding="utf-8")
    result = validate_npu_strict(configured_bundle)
    assert not result.passed
    assert next(check for check in result.checks if check.name == "genai_config_parse").status == "FAIL"
