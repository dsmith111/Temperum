"""The search space: which tensor classes of a GGUF get which quantization type.

A candidate is a base type for everything plus overrides for named tensor classes, expressed as the
llama-quantize flags that build it (`--tensor-type attn_v=q8_0`), so a recipe is reproducible with the
quantizer alone. The head (token embedding, or the output tensor when untied) is left to the quantizer's
own rule on purpose: for Q4_0 it already keeps a tied head at Q6_K, and measured upgrades of it changed
nothing (measured 2026-10-22).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# the projection classes of a transformer block, as llama.cpp names its tensors
CLASSES: tuple[str, ...] = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
# types the Hexagon backend runs on the HTP; the first space upgrades classes from the base to Q8_0 only,
# because every other type measured a kernel penalty on decode beyond its bytes (measured 2026-10-22)
BASE_TYPE = "Q4_0"
UPGRADE_TYPE = "q8_0"


@dataclass(frozen=True)
class Candidate:
    """A quantization mix: `base` for every tensor, `overrides` (class -> ggml type) on top."""

    base: str = BASE_TYPE
    overrides: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @classmethod
    def plain(cls, base: str = BASE_TYPE) -> Candidate:
        return cls(base, ())

    @classmethod
    def upgrade(cls, classes: tuple[str, ...] | list[str], to: str = UPGRADE_TYPE, base: str = BASE_TYPE) -> Candidate:
        return cls(base, tuple((c, to) for c in sorted(classes, key=CLASSES.index)))

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(c for c, _t in self.overrides)

    @property
    def name(self) -> str:
        """Stable, file-system-safe: `q4_0` or `q4_0+attn_v.q8_0+ffn_up.q8_0`."""
        return self.base.lower() + "".join(f"+{c}.{t}" for c, t in self.overrides)

    @property
    def label(self) -> str:
        """For people: `Q4_0` or `Q4_0, attn_v and ffn_up at Q8_0`."""
        if not self.overrides:
            return self.base
        by_type: dict[str, list[str]] = {}
        for c, t in self.overrides:
            by_type.setdefault(t.upper(), []).append(c)
        parts = [f"{_join(cs)} at {t}" for t, cs in by_type.items()]
        return f"{self.base}, {'; '.join(parts)}"

    def quantize_args(self) -> list[str]:
        """The llama-quantize flags that build this mix (before the input and output paths)."""
        args: list[str] = []
        for c, t in self.overrides:
            args += ["--tensor-type", f"{c}={t}"]
        return args

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "base": self.base, "overrides": dict(self.overrides),
                "quantize_args": self.quantize_args()}


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]
