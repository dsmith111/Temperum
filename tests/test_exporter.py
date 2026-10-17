from __future__ import annotations

from pathlib import Path

import pytest

from npu_model.core.errors import NpuModelError
from npu_model.exporters.ort_genai_builder import OrtGenaiBuilderExporter
from npu_model.exporters.ort_genai_builder import resolve_export_precision
from npu_model.core.types import ModelInfo


def test_exporter_has_id() -> None:
    e = OrtGenaiBuilderExporter()
    assert e.id == "ort-genai-builder"


def test_can_export_phi3() -> None:
    e = OrtGenaiBuilderExporter()
    mi = ModelInfo(
        source={}, model_type="phi3", architectures=[], config={}, tokenizer_files=[]
    )
    assert e.can_export(mi) is True


def test_can_export_llama() -> None:
    e = OrtGenaiBuilderExporter()
    mi = ModelInfo(
        source={}, model_type="llama", architectures=[], config={}, tokenizer_files=[]
    )
    assert e.can_export(mi) is True


def test_cannot_export_unknown() -> None:
    e = OrtGenaiBuilderExporter()
    mi = ModelInfo(
        source={}, model_type="unknown_arch", architectures=[], config={}, tokenizer_files=[]
    )
    assert e.can_export(mi) is False


def test_check_dependencies_reports_missing() -> None:
    e = OrtGenaiBuilderExporter()
    missing = e.check_dependencies()
    # In dev env, at minimum onnxruntime_genai is not installed
    assert isinstance(missing, list)
    # We can't assert exact contents since torch/transformers may or may not be present


@pytest.mark.parametrize("precision", ["bf16", "int8", "", "FP16"])
def test_unsupported_precision_is_rejected_before_dependency_checks(
    precision: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def should_not_check_deps(self: OrtGenaiBuilderExporter) -> list[str]:
        pytest.fail("Unsupported precision must be rejected before export/dependency probing")

    monkeypatch.setattr(OrtGenaiBuilderExporter, "check_dependencies", should_not_check_deps)
    with pytest.raises(NpuModelError) as exc:
        OrtGenaiBuilderExporter().export(
            tmp_path, tmp_path / "out",
            ModelInfo({}, "phi3", [], {}, []),
            export_config={"precision": precision},
        )
    assert exc.value.reason_code == "UNSUPPORTED_EXPORT_PRECISION"
    assert not (tmp_path / "out").exists()


def test_explicit_int4_is_not_silently_replaced_before_qdq() -> None:
    with pytest.raises(NpuModelError) as exc:
        resolve_export_precision("int4", "qnn-qdq")
    assert exc.value.reason_code == "INCOMPATIBLE_EXPORT_PRECISION"


def test_omitted_precision_chooses_truthful_stage_default() -> None:
    assert resolve_export_precision(None) == "int4"
    assert resolve_export_precision(None, "qnn-qdq") == "fp16"
    assert resolve_export_precision("fp32", "qnn-qdq") == "fp32"
