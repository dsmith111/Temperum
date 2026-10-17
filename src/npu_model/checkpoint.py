"""Tensor names and sizes from safetensors headers, without loading any weights."""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from npu_model.core.errors import NpuModelError

# Repeated transformer blocks: model.layers.N (Llama/Qwen/Phi/Gemma), transformer.h.N (GPT-2
# style), *.blocks.N.
_BLOCK_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.\d+\.")
_MAX_HEADER = 100 * 1024 * 1024


@dataclass(frozen=True)
class CheckpointInfo:
    tensors: dict[str, int]  # tensor name -> element count

    @property
    def modules(self) -> frozenset[str]:
        """Names of modules that own a weight tensor (e.g. model.layers.0.self_attn.v_proj)."""
        return frozenset(name[: -len(".weight")] for name in self.tensors if name.endswith(".weight"))

    @property
    def block_params(self) -> int:
        """Parameters inside the repeated transformer blocks (what the NPU graphs hold)."""
        return sum(count for name, count in self.tensors.items() if _BLOCK_RE.search(name))

    @property
    def total_params(self) -> int:
        return sum(self.tensors.values())


def _read_header(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("file is shorter than a safetensors header")
        (length,) = struct.unpack("<Q", raw)
        if length > _MAX_HEADER:
            raise ValueError(f"implausible header length {length}")
        header = json.loads(handle.read(length).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("header is not a JSON object")
    return header


def read_checkpoint(model_dir: Path) -> CheckpointInfo:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise NpuModelError("source", "NO_SAFETENSORS", f"No .safetensors weights in {model_dir}")
    tensors: dict[str, int] = {}
    for path in files:
        try:
            header = _read_header(path)
        except (OSError, ValueError) as exc:
            raise NpuModelError("source", "BAD_SAFETENSORS", f"Cannot read {path.name}: {exc}",
                                hint="Delete the cached snapshot and fetch it again.") from exc
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            tensors[name] = math.prod(int(dim) for dim in meta.get("shape", []))
    return CheckpointInfo(tensors)
