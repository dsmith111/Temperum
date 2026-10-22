"""What "best" means: constraints first, then quality, with measurement noise treated as ties.

The default is the rule experiment 1 was decided by: a candidate must sit whole on the NPU and decode at
no less than 90% of the plain base; among those the lowest perplexity ratio wins, and ratios within the
noise of the measurement are ties that go to the smaller file, then the faster decode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from temperum.optimize.evaluate import Trial


@dataclass(frozen=True)
class Objective:
    decode_cap: float = 0.10       # largest decode loss against the plain base, as a fraction
    # Measured on the same source twice (Qwen2.5-1.5B, 8 candidates): perplexity repeats exactly, since the
    # chunks and the weights are the same bytes, so ratio_noise is about what a 16x512-token sample can tell
    # apart, not measurement scatter; decode repeated within 1.5% on average and 3.6% at most.
    ratio_noise: float = 0.02      # ratio differences up to this are ties (16 chunks of 512 tokens)
    decode_noise: float = 0.05     # decode differences up to this fraction are ties
    size_budget_mib: float | None = None

    def admissible(self, trial: Trial, base: Trial) -> bool:
        if trial.error or trial.ratio is None or trial.decode is None or not trial.whole:
            return False
        if base.decode and trial.decode < base.decode * (1.0 - self.decode_cap) * (1.0 - self.decode_noise):
            return False
        return self.size_budget_mib is None or trial.size_mib <= self.size_budget_mib

    def choose(self, trials: list[Trial], base: Trial) -> Trial:
        """The best admissible trial; the base itself when nothing beats it beyond noise."""
        best = base
        for t in trials:
            if t is base or not self.admissible(t, base):
                continue
            if self.better(t, best):
                best = t
        return best

    def better(self, a: Trial, b: Trial) -> bool:
        """a beats b: clearly lower ratio; or a tie on ratio and smaller; or a tie on both and faster."""
        assert a.ratio is not None and b.ratio is not None
        if b.ratio - a.ratio > self.ratio_noise:
            return True
        if a.ratio - b.ratio > self.ratio_noise:
            return False
        if a.size_mib < b.size_mib:
            return True
        if a.size_mib > b.size_mib:
            return False
        return bool(a.decode and b.decode and a.decode > b.decode * (1.0 + self.decode_noise))

    def to_dict(self) -> dict[str, Any]:
        return {"decode_cap": self.decode_cap, "ratio_noise": self.ratio_noise, "decode_noise": self.decode_noise,
                "size_budget_mib": self.size_budget_mib,
                "rule": "whole on the NPU; decode within the cap of plain Q4_0; then the lowest perplexity ratio; "
                        "ties within noise go to the smaller file, then the faster decode"}
