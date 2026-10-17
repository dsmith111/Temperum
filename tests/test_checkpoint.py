from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from npu_model.checkpoint import read_checkpoint
from npu_model.core.errors import NpuModelError


def _write_safetensors(path: Path, tensors: dict[str, list[int]]) -> None:
    header = {name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]} for name, shape in tensors.items()}
    header["__metadata__"] = {"format": "pt"}  # type: ignore[dict-item]
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def test_reads_names_and_sizes_across_shards(tmp_path: Path) -> None:
    _write_safetensors(tmp_path / "model-00001-of-00002.safetensors", {
        "model.embed_tokens.weight": [100, 8],
        "model.layers.0.self_attn.v_proj.weight": [8, 8],
        "model.layers.0.self_attn.v_proj.bias": [8],
    })
    _write_safetensors(tmp_path / "model-00002-of-00002.safetensors", {
        "model.layers.1.mlp.down_proj.weight": [8, 16],
        "lm_head.weight": [100, 8],
    })
    info = read_checkpoint(tmp_path)
    assert info.total_params == 800 + 64 + 8 + 128 + 800
    assert info.block_params == 64 + 8 + 128
    assert info.modules == {"model.embed_tokens", "model.layers.0.self_attn.v_proj",
                            "model.layers.1.mlp.down_proj", "lm_head"}


def test_missing_or_corrupt_weights_are_reported(tmp_path: Path) -> None:
    with pytest.raises(NpuModelError) as exc:
        read_checkpoint(tmp_path)
    assert exc.value.reason_code == "NO_SAFETENSORS"
    (tmp_path / "model.safetensors").write_bytes(b"\x00\x01")
    with pytest.raises(NpuModelError) as exc:
        read_checkpoint(tmp_path)
    assert exc.value.reason_code == "BAD_SAFETENSORS"
