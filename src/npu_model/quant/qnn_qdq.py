"""QNN QDQ quantizer — produces quantized ONNX graphs for QNN HTP execution.

Uses ORT's QNN-specific quantization pipeline:
  1. ``qnn_preprocess_model()``  — prepares the graph for QNN quantization
  2. ``get_qnn_qdq_config()``   — generates QDQ config using calibration data
  3. ``quantize()``             — applies QDQ quantization

This is the documented flow for QNN HTP. Dynamic quantization is NOT
supported for HTP — calibration data is required.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from npu_model.core.errors import NpuModelError
from npu_model.core.types import GraphBundle


@dataclass
class QnnQdqQuantizer:
    """Quantize ONNX graphs using ORT's QNN QDQ pipeline.

    Requires calibration data — will fail with a clear error if none provided.
    Heavy dependencies imported lazily.
    """
    id: str = "qnn-qdq"
    requires_calibration: bool = True

    def apply(self, graphs: GraphBundle, *, quant_config: dict[str, Any]) -> GraphBundle:
        try:
            import_module("onnxruntime")
        except ImportError:
            raise NpuModelError(
                stage="quant",
                reason_code="MISSING_QUANT_DEPS",
                message="onnxruntime is required for QNN QDQ quantization",
                hint="pip install onnxruntime (x64 build recommended for quantization)",
            )

        # Import QNN-specific quantization helpers
        try:
            from onnxruntime.quantization import quantize
            from onnxruntime.quantization.execution_providers.qnn import (
                get_qnn_qdq_config,
                qnn_preprocess_model,
            )
        except ImportError as e:
            raise NpuModelError(
                stage="quant", reason_code="MISSING_QNN_QUANT_HELPERS",
                message=f"ORT QNN quantization helpers are unavailable: {e}",
                hint="Install an ORT build with QNN quantization helpers; generic QDQ is not a substitute.",
                cause=e,
            ) from e

        calib_data_reader = quant_config.get("calibration_data_reader")

        # HARD REQUIREMENT: calibration data is required for QNN HTP
        if calib_data_reader is None:
            raise NpuModelError(
                stage="quant",
                reason_code="NO_CALIBRATION_DATA",
                message="QNN QDQ quantization requires calibration data for HTP.",
                hint=(
                    "Provide calibration data via --calib-prompts <file>.\n"
                    "Dynamic quantization is NOT supported for QNN HTP.\n"
                    "For prebuilt models that are already quantized, use --quant passthrough."
                ),
            )

        new_graphs: dict[str, Path] = {}
        for name, graph_path in graphs.graphs.items():
            out_dir = graph_path.parent / f"{name}_qdq"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Ensure external data file is co-located with the graph
            data_file = graph_path.parent / f"{graph_path.name}.data"
            if data_file.exists():
                dst_data = out_dir / data_file.name
                if not dst_data.exists():
                    shutil.copy2(data_file, dst_data)
                work_graph = out_dir / graph_path.name
                if not work_graph.exists():
                    shutil.copy2(graph_path, work_graph)
                graph_path = work_graph

            has_external_data = any(out_dir.glob("*.data"))
            qdq_path = out_dir / f"{name}.qdq.onnx"

            qdq_path = self._quantize_qnn_flow(
                name=name,
                graph_path=graph_path,
                out_dir=out_dir,
                calib_data_reader=calib_data_reader,
                has_external_data=has_external_data,
                qnn_preprocess_model=qnn_preprocess_model,
                get_qnn_qdq_config=get_qnn_qdq_config,
                quantize_fn=quantize,
                quant_config=quant_config,
            )

            if not qdq_path.exists():
                raise NpuModelError(
                    stage="quant",
                    reason_code="NO_QUANT_OUTPUT",
                    message=f"Quantization produced no output for '{name}'",
                )

            new_graphs[name] = qdq_path

        return GraphBundle(
            graphs=new_graphs,
            tokenizer_dir=graphs.tokenizer_dir,
            extra_files=graphs.extra_files,
            metadata={**graphs.metadata, "quantizer": self.id},
        )

    def _quantize_qnn_flow(
        self,
        *,
        name: str,
        graph_path: Path,
        out_dir: Path,
        calib_data_reader: Any,
        has_external_data: bool,
        qnn_preprocess_model: Any,
        get_qnn_qdq_config: Any,
        quantize_fn: Any,
        quant_config: dict[str, Any],
    ) -> Path:
        """QNN-specific quantization: preprocess → get_qnn_qdq_config → quantize."""
        from onnxruntime.quantization import QuantType

        # Step 1: QNN preprocessing
        pp_path = out_dir / f"{name}_preprocessed.onnx"
        try:
            model_changed = bool(
                qnn_preprocess_model(
                    model_input=str(graph_path),
                    model_output=str(pp_path),
                )
            )
        except Exception as e:
            raise NpuModelError(
                stage="quant",
                reason_code="QNN_PREPROCESS_FAILED",
                message=f"qnn_preprocess_model failed for '{name}': {e}",
                hint="The model may have unsupported ops for QNN preprocessing.",
                cause=e,
            ) from e

        # Use the preprocessed model only when preprocessing actually changed it;
        # otherwise keep using the original graph (ORT may not emit a file).
        model_to_quantize = pp_path if model_changed else graph_path

        if model_changed and not pp_path.exists():
            raise NpuModelError(
                stage="quant",
                reason_code="QNN_PREPROCESS_OUTPUT_MISSING",
                message=f"qnn_preprocess_model reported changes for '{name}' but did not create: {pp_path}",
            )

        # Copy external data alongside preprocessed model
        if model_changed and has_external_data:
            for data in out_dir.glob("*.data"):
                dst = pp_path.parent / data.name
                if not dst.exists():
                    shutil.copy2(data, dst)

        # Step 2: Get QNN QDQ config
        qdq_path = out_dir / f"{name}.qdq.onnx"
        try:
            qdq_config = get_qnn_qdq_config(
                model_input=str(model_to_quantize),
                calibration_data_reader=calib_data_reader,
                activation_type=quant_config.get("activation_type", QuantType.QUInt16),
                weight_type=quant_config.get("weight_type", QuantType.QUInt8),
            )
        except Exception as e:
            raise NpuModelError(
                stage="quant",
                reason_code="QNN_QDQ_CONFIG_FAILED",
                message=f"get_qnn_qdq_config failed for '{name}': {e}",
                hint=f"Model used for QDQ config: {model_to_quantize}",
                cause=e,
            ) from e

        # Step 3: Quantize
        try:
            quantize_fn(
                model_input=str(model_to_quantize),
                model_output=str(qdq_path),
                quant_config=qdq_config,
            )
        except Exception as e:
            raise NpuModelError(
                stage="quant",
                reason_code="QUANTIZATION_FAILED",
                message=f"QDQ quantization failed for graph '{name}': {e}",
                hint=f"Model used for quantization: {model_to_quantize}",
                cause=e,
            ) from e

        return qdq_path

    def _quantize_fallback(
        self,
        *,
        name: str,
        graph_path: Path,
        qdq_path: Path,
        calib_data_reader: Any,
        has_external_data: bool,
        quant_config: dict[str, Any],
    ) -> Path:
        """Legacy generic QDQ helper; never a substitute selected by qnn-qdq."""
        from onnxruntime.quantization import QuantType, quantize_static

        if calib_data_reader is None:
            raise NpuModelError(
                stage="quant",
                reason_code="NO_CALIBRATION_DATA",
                message=f"Calibration data required for QDQ quantization of '{name}'.",
                hint="Provide --calib-prompts. Dynamic quantization is not supported for HTP.",
            )

        try:
            quantize_static(
                model_input=str(graph_path),
                model_output=str(qdq_path),
                calibration_data_reader=calib_data_reader,
                quant_format=3,  # QDQ
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                op_types_to_quantize=quant_config.get("op_types", None),
                extra_options=quant_config.get("extra_options", {}),
                use_external_data_format=has_external_data,
            )
        except Exception as e:
            raise NpuModelError(
                stage="quant",
                reason_code="QUANTIZATION_FAILED",
                message=f"QDQ quantization failed for graph '{name}': {e}",
                hint="Check model compatibility with ORT quantization pipeline.",
                cause=e,
            ) from e

        return qdq_path
