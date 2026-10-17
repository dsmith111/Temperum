from __future__ import annotations

from pathlib import Path
import sys
from enum import Enum
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from npu_model.core.types import GraphBundle
from npu_model.quant.passthrough import PassthroughQuantizer


def test_passthrough_is_noop(tmp_path: Path) -> None:
    q = PassthroughQuantizer()
    bundle = GraphBundle(
        graphs={"m": tmp_path / "m.onnx"},
        tokenizer_dir=tmp_path,
        extra_files=[],
        metadata={},
    )
    assert q.apply(bundle, quant_config={}) is bundle


def test_passthrough_has_id() -> None:
    q = PassthroughQuantizer()
    assert q.id == "passthrough"


def test_qnn_qdq_has_id() -> None:
    from npu_model.quant.qnn_qdq import QnnQdqQuantizer
    q = QnnQdqQuantizer()
    assert q.id == "qnn-qdq"


class TestQnnQdqPreprocessBranching:
    """Control-flow regressions with mocked ORT, not QNN supplier qualification."""

    @pytest.fixture(autouse=True)
    def mock_quant_types(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class QuantType(Enum):
            QUInt16 = 1
            QUInt8 = 2

        module = ModuleType("onnxruntime.quantization")
        module.QuantType = QuantType
        monkeypatch.setitem(sys.modules, "onnxruntime.quantization", module)

    def _make_graph(self, tmp_path: Path) -> tuple[Path, Path]:
        graph = tmp_path / "model.onnx"
        graph.write_bytes(b"fake-onnx")
        out_dir = tmp_path / "model_qdq"
        out_dir.mkdir()
        return graph, out_dir

    def test_no_preprocess_rewrite_uses_original_path(self, tmp_path: Path) -> None:
        """When qnn_preprocess_model returns False, the original graph_path
        must be passed to get_qnn_qdq_config and quantize — not the
        non-existent preprocessed path."""
        from npu_model.quant.qnn_qdq import QnnQdqQuantizer

        graph_path, out_dir = self._make_graph(tmp_path)
        qdq_path = out_dir / "model.qdq.onnx"

        mock_preprocess = MagicMock(return_value=False)
        mock_config = MagicMock(return_value="fake-config")
        mock_quantize = MagicMock(side_effect=lambda **kw: Path(kw["model_output"]).write_bytes(b"q"))
        mock_calib = MagicMock()

        q = QnnQdqQuantizer()
        result = q._quantize_qnn_flow(
            name="model",
            graph_path=graph_path,
            out_dir=out_dir,
            calib_data_reader=mock_calib,
            has_external_data=False,
            qnn_preprocess_model=mock_preprocess,
            get_qnn_qdq_config=mock_config,
            quantize_fn=mock_quantize,
            quant_config={},
        )

        # get_qnn_qdq_config and quantize must use the original graph_path
        mock_config.assert_called_once()
        assert mock_config.call_args[1]["model_input"] == str(graph_path)

        mock_quantize.assert_called_once()
        assert mock_quantize.call_args[1]["model_input"] == str(graph_path)
        assert result == qdq_path
        assert result.is_file()

        # The preprocessed file should NOT have been created
        pp_path = out_dir / "model_preprocessed.onnx"
        assert not pp_path.exists()

    def test_preprocess_rewrite_uses_preprocessed_path(self, tmp_path: Path) -> None:
        """When qnn_preprocess_model returns True and creates the file,
        the preprocessed path must be passed downstream."""
        from npu_model.quant.qnn_qdq import QnnQdqQuantizer

        graph_path, out_dir = self._make_graph(tmp_path)
        pp_path = out_dir / "model_preprocessed.onnx"

        def fake_preprocess(**kw):
            Path(kw["model_output"]).write_bytes(b"preprocessed")
            return True

        mock_preprocess = MagicMock(side_effect=fake_preprocess)
        mock_config = MagicMock(return_value="fake-config")
        mock_quantize = MagicMock(side_effect=lambda **kw: Path(kw["model_output"]).write_bytes(b"q"))
        mock_calib = MagicMock()

        q = QnnQdqQuantizer()
        q._quantize_qnn_flow(
            name="model",
            graph_path=graph_path,
            out_dir=out_dir,
            calib_data_reader=mock_calib,
            has_external_data=False,
            qnn_preprocess_model=mock_preprocess,
            get_qnn_qdq_config=mock_config,
            quantize_fn=mock_quantize,
            quant_config={},
        )

        mock_config.assert_called_once()
        assert mock_config.call_args[1]["model_input"] == str(pp_path)

        mock_quantize.assert_called_once()
        assert mock_quantize.call_args[1]["model_input"] == str(pp_path)

    def test_preprocess_changed_but_file_missing_raises(self, tmp_path: Path) -> None:
        """When qnn_preprocess_model returns True but doesn't create the file,
        a clear error must be raised."""
        from npu_model.core.errors import NpuModelError
        from npu_model.quant.qnn_qdq import QnnQdqQuantizer

        graph_path, out_dir = self._make_graph(tmp_path)

        mock_preprocess = MagicMock(return_value=True)

        q = QnnQdqQuantizer()
        with pytest.raises(NpuModelError) as exc_info:
            q._quantize_qnn_flow(
                name="model",
                graph_path=graph_path,
                out_dir=out_dir,
                calib_data_reader=MagicMock(),
                has_external_data=False,
                qnn_preprocess_model=mock_preprocess,
                get_qnn_qdq_config=MagicMock(),
                quantize_fn=MagicMock(),
                quant_config={},
            )
        assert exc_info.value.reason_code == "QNN_PREPROCESS_OUTPUT_MISSING"


def test_missing_qnn_helpers_cannot_select_generic_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npu_model.core.errors import NpuModelError
    from npu_model.quant.qnn_qdq import QnnQdqQuantizer

    monkeypatch.setitem(sys.modules, "onnxruntime", ModuleType("onnxruntime"))
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", None)
    graphs = GraphBundle({"model": tmp_path / "model.onnx"}, tmp_path, [], {})
    q = QnnQdqQuantizer()
    with pytest.raises(NpuModelError) as exc:
        q.apply(graphs, quant_config={"calibration_data_reader": object()})
    assert exc.value.reason_code == "MISSING_QNN_QUANT_HELPERS"


def test_missing_ort_reports_quant_dependency_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npu_model.core.errors import NpuModelError
    from npu_model.quant.qnn_qdq import QnnQdqQuantizer

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(NpuModelError) as exc:
        QnnQdqQuantizer().apply(GraphBundle({}, tmp_path, [], {}), quant_config={})
    assert exc.value.reason_code == "MISSING_QUANT_DEPS"
