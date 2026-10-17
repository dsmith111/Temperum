"""Olive-backed QNN LLM quantizer.

This is the supported route for decoder-only LLM families on QNN HTP.
It generates a family-specific Olive config, runs Olive externally, then maps
produced artifacts back into the internal GraphBundle contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

from npu_model.core.errors import NpuModelError
from npu_model.core.types import GraphBundle
from npu_model.olive.artifacts import collect_olive_outputs
from npu_model.olive.compat import probe_olive_python
from npu_model.olive.config_builder import (
    SUPPORTED_OLIVE_LLM_FAMILIES,
    build_olive_config,
)
from npu_model.olive.runner import run_olive_cli


@dataclass
class OliveQnnLlmQuantizer:
    """Olive-backed LLM quantizer for QNN HTP deployment.

    Production route for decoder-only LLMs (Phi / Phi-3 / Llama).
    Requires: ``pip install olive-ai[auto-opt]``
    """

    id: str = "olive-qnn-llm"
    requires_calibration: bool = False

    SUPPORTED_FAMILIES = SUPPORTED_OLIVE_LLM_FAMILIES

    def apply(self, graphs: GraphBundle, *, quant_config: dict[str, Any]) -> GraphBundle:
        if not graphs.graphs:
            raise NpuModelError(
                stage="quant",
                reason_code="OLIVE_INPUT_EMPTY",
                message="No input ONNX graphs were provided for olive-qnn-llm.",
            )

        olive_python_raw = (
            quant_config.get("olive_python")
            or os.environ.get("NPU_MODEL_OLIVE_PYTHON")
            or sys.executable
        )
        olive_python = Path(str(olive_python_raw))
        report = probe_olive_python(olive_python)

        import logging
        _log = logging.getLogger("npu_model")
        if report.olive_version:
            _log.info("Using external Olive interpreter: %s (olive-ai %s)", report.python_exe, report.olive_version)
        if not report.is_x64:
            _log.warning(
                "Olive interpreter is not x64 (%s). Quantization is recommended on x64 hosts.",
                report.machine,
            )

        # Log input model routing decision
        model_info = quant_config.get("model_info") or {}
        source = (model_info.get("source") or {})
        if source.get("type") == "hf" and quant_config.get("model_dir"):
            _log.info(
                "Olive will use HfModel input (HF source: %s, local: %s). "
                "Adapter-exported ONNX will be bypassed in favor of Olive's LLM-aware export.",
                source.get("repo_id", "unknown"),
                quant_config["model_dir"],
            )

        first_graph = next(iter(graphs.graphs.values()))
        work_dir = Path(first_graph).parent / "_olive_qnn_llm"

        plan = build_olive_config(graphs=graphs, quant_config=quant_config, work_dir=work_dir)
        run_olive_cli(
            python_exe=report.python_exe,
            config_path=plan.config_path,
            work_dir=work_dir,
            timeout_s=int(quant_config.get("olive_timeout_s", 14_400)),
        )
        return collect_olive_outputs(
            olive_output_dir=plan.output_dir,
            fallback_tokenizer_dir=graphs.tokenizer_dir,
            fallback_extra_files=graphs.extra_files,
            family=plan.family,
        )
