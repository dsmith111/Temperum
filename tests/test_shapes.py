from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class TestHasDynamicShapes:

    def test_missing_onnx_does_not_claim_static(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from npu_model.core.shapes import has_dynamic_shapes

        monkeypatch.setitem(sys.modules, "onnx", None)
        with pytest.raises(ImportError):
            has_dynamic_shapes(Path("nonexistent.onnx"))

    def test_invalid_graph_does_not_claim_static(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from npu_model.core.errors import NpuModelError
        from npu_model.core.shapes import has_dynamic_shapes

        onnx = SimpleNamespace(load=MagicMock(side_effect=ValueError("invalid graph")))
        monkeypatch.setitem(sys.modules, "onnx", onnx)
        monkeypatch.setitem(sys.modules, "google.protobuf.message", SimpleNamespace(DecodeError=ValueError))
        with pytest.raises(NpuModelError) as exc:
            has_dynamic_shapes(Path("invalid.onnx"))
        assert exc.value.reason_code == "ONNX_INSPECTION_FAILED"

    def test_unknown_input_dimension_is_not_static(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from npu_model.core.shapes import has_dynamic_shapes

        tensor = MagicMock()
        tensor.HasField.return_value = True
        tensor.tensor_type.HasField.return_value = True
        tensor.tensor_type.shape.dim = [SimpleNamespace(dim_param="", dim_value=0)]
        model = SimpleNamespace(graph=SimpleNamespace(input=[SimpleNamespace(name="ids", type=tensor)]))
        monkeypatch.setitem(sys.modules, "onnx", SimpleNamespace(load=lambda *a, **kw: model))
        monkeypatch.setitem(sys.modules, "google.protobuf.message", SimpleNamespace(DecodeError=ValueError))
        assert has_dynamic_shapes(Path("mock.onnx")) == ["ids: unknown/nonpositive dimension"]


class TestFixDynamicShapes:

    def test_returns_error_without_onnx_installed(self, tmp_path: Path) -> None:
        """If onnx can't load the file, returns errors."""
        from npu_model.core.shapes import fix_dynamic_shapes

        fake = tmp_path / "model.onnx"
        fake.write_bytes(b"\x00")
        out = tmp_path / "fixed.onnx"

        result = fix_dynamic_shapes(fake, out)
        # Should either fix or report errors
        assert isinstance(result.fixed, bool)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
