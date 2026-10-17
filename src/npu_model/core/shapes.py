"""Fix dynamic shapes in ONNX models for QNN/HTP compatibility.

QNN HTP requires all tensor dimensions to be static (fixed at graph-compile
time).  This module inspects ONNX graphs for dynamic dimensions and either
fixes them (by setting concrete values) or reports what prevents fixing.

Typical fixable dimensions:
  - batch_size → 1
  - sequence_length / past_sequence_length → num_ctx
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from npu_model.core.errors import NpuModelError


@dataclass
class ShapeFixResult:
    fixed: bool
    output_path: Path | None
    fixed_dims: dict[str, int]       # dim_name → concrete value
    errors: list[str]
    warnings: list[str]


def fix_dynamic_shapes(
    onnx_path: Path,
    out_path: Path,
    *,
    batch_size: int = 1,
    sequence_length: int = 512,
    extra_dims: dict[str, int] | None = None,
) -> ShapeFixResult:
    """Attempt to fix dynamic dimensions in an ONNX model.

    Uses onnx shape inference + manual dim overrides.
    Heavy deps (onnx) are imported lazily.
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        import onnx
    except ImportError:
        return ShapeFixResult(
            fixed=False, output_path=None,
            fixed_dims={}, errors=["onnx package not installed"], warnings=[],
        )

    try:
        model = onnx.load(str(onnx_path), load_external_data=False)
    except Exception as e:
        return ShapeFixResult(
            fixed=False, output_path=None,
            fixed_dims={}, errors=[f"Failed to load ONNX model: {e}"], warnings=[],
        )

    # Common dynamic dim names → concrete values
    #
    # NOTE: past_sequence_length is mapped to sequence_length here for
    # compile-time shape fixing (QNN HTP requires fully static shapes).
    # For calibration, this means KV-cache tensors will be sized at
    # num_ctx, which may use significant memory.  A future improvement
    # could separate calibration-time shape policy (small past, e.g. 1)
    # from compile-time context sizing.
    dim_map: dict[str, int] = {
        "batch_size": batch_size,
        "batch": batch_size,
        "N": batch_size,
        "sequence_length": sequence_length,
        "seq_len": sequence_length,
        "past_sequence_length": sequence_length,
        "past_seq_len": sequence_length,
        "total_sequence_length": sequence_length,
        "max_sequence_length": sequence_length,
    }
    if extra_dims:
        dim_map.update(extra_dims)

    fixed_dims: dict[str, int] = {}

    # Fix inputs
    for inp in model.graph.input:
        if inp.type.HasField("tensor_type"):
            shape = inp.type.tensor_type.shape
            if shape is not None:
                for dim in shape.dim:
                    if dim.dim_param:  # symbolic / dynamic
                        name = dim.dim_param
                        if name in dim_map:
                            dim.dim_value = dim_map[name]
                            dim.ClearField("dim_param")
                            fixed_dims[name] = dim_map[name]
                        else:
                            warnings.append(
                                f"Input '{inp.name}': unfixed dynamic dim '{name}'. "
                                f"Pass --shape-dim {name}=<value> to fix it."
                            )

    # Fix outputs
    for out in model.graph.output:
        if out.type.HasField("tensor_type"):
            shape = out.type.tensor_type.shape
            if shape is not None:
                for dim in shape.dim:
                    if dim.dim_param:
                        name = dim.dim_param
                        if name in dim_map:
                            dim.dim_value = dim_map[name]
                            dim.ClearField("dim_param")
                            fixed_dims[name] = dim_map[name]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy co-located external data file if present, so the saved model
    # can reference it at the new location.
    import shutil
    data_file = onnx_path.parent / f"{onnx_path.name}.data"
    if data_file.exists():
        dst_data = out_path.parent / f"{out_path.name}.data"
        if not dst_data.exists():
            shutil.copy2(data_file, dst_data)

    try:
        onnx.save(model, str(out_path))
    except Exception as e:
        errors.append(f"Failed to save fixed model: {e}")
        return ShapeFixResult(
            fixed=False, output_path=None,
            fixed_dims=fixed_dims, errors=errors, warnings=warnings,
        )

    return ShapeFixResult(
        fixed=True,
        output_path=out_path,
        fixed_dims=fixed_dims,
        errors=errors,
        warnings=warnings,
    )


def has_dynamic_shapes(onnx_path: Path) -> list[str]:
    """Inspect input dimensions; missing dependencies or unreadable graphs are not static."""
    import onnx
    from google.protobuf.message import DecodeError

    try:
        model = onnx.load(str(onnx_path), load_external_data=False)
    except (OSError, ValueError, RuntimeError, DecodeError) as e:
        raise NpuModelError(
            stage="shapes", reason_code="ONNX_INSPECTION_FAILED",
            message=f"Could not inspect input shapes in {onnx_path.name}: {e}",
            cause=e,
        ) from e

    dynamic_dims: list[str] = []
    for inp in model.graph.input:
        if inp.type.HasField("tensor_type"):
            if not inp.type.tensor_type.HasField("shape"):
                dynamic_dims.append(f"{inp.name}: unknown rank")
                continue
            shape = inp.type.tensor_type.shape
            if shape is not None:
                for dim in shape.dim:
                    if dim.dim_param and dim.dim_param not in dynamic_dims:
                        dynamic_dims.append(dim.dim_param)
                    elif not dim.dim_param and dim.dim_value <= 0:
                        dynamic_dims.append(f"{inp.name}: unknown/nonpositive dimension")
        else:
            dynamic_dims.append(f"{inp.name}: non-tensor input")
    return dynamic_dims
