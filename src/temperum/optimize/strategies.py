"""How the space is searched.

`structured` is the strategy experiment 1 validated: measure the plain base and each tensor class alone
at Q8_0 (one trial per class), fit an additive surrogate (log-perplexity deltas add; decode falls with the
bytes moved off the base, linearly), enumerate every composition on the surrogate under the objective's
decode cap, measure the predicted best, and let the objective choose among what was measured. On
Qwen2.5-1.5B the surrogate placed three independently measured compositions within 0.2% of their
measured perplexity and decode within the measurement's own noise (CONVERTER_PLAN.md, D22).

`listed` measures an explicit list of candidates: a grid for small spaces, or a user's own shortlist.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from temperum.optimize.evaluate import Evaluator, Trial
from temperum.optimize.objective import Objective
from temperum.optimize.space import CLASSES, Candidate

Progress = Callable[[str], None]


@dataclass
class Surrogate:
    """Per-class effects measured alone, and the decode cost of bytes beyond the base."""

    base: Trial
    effects: dict[str, tuple[float, float]]  # class -> (delta log ppl, extra MiB)
    slope: float                              # tok/s lost per extra MiB
    rms: float                                # residual of the decode fit, tok/s

    def predict(self, classes: tuple[str, ...]) -> dict[str, float]:
        dlog = sum(self.effects[c][0] for c in classes)
        extra = sum(self.effects[c][1] for c in classes)
        assert self.base.ppl is not None and self.base.decode is not None
        return {"ppl": self.base.ppl * math.exp(dlog), "size_mib": self.base.size_mib + extra,
                "decode": self.base.decode - self.slope * extra}

    def to_dict(self) -> dict[str, Any]:
        return {"effects": {c: {"dlog_ppl": round(d, 4), "extra_mib": round(m, 1)} for c, (d, m) in self.effects.items()},
                "decode_tps_per_mib": round(self.slope, 5), "decode_fit_rms_tps": round(self.rms, 2)}


@dataclass
class SearchResult:
    base: Trial
    trials: list[Trial]
    chosen: Trial
    strategy: str
    reference_ppl: float
    surrogate: Surrogate | None = None
    predicted: dict[str, Any] = field(default_factory=dict)


def fit_surrogate(base: Trial, singles: dict[str, Trial], anchors: list[Trial] | None = None) -> Surrogate:
    """`anchors`: trials with no class of their own that still pin the decode fit, such as the Q8_0 source:
    the singles move decode by less than its measurement noise, so a far point sets the slope."""
    assert base.ppl is not None and base.decode is not None
    effects = {c: (math.log(t.ppl / base.ppl), t.size_mib - base.size_mib)
               for c, t in singles.items() if t.ppl is not None and t.error is None}
    points = [(t.size_mib - base.size_mib, base.decode - t.decode)
              for t in [*singles.values(), *(anchors or [])] if t.decode is not None and t.error is None]
    den = sum(x * x for x, _y in points)
    slope = max(0.0, sum(x * y for x, y in points) / den) if den else 0.0
    resid = [y - slope * x for x, y in points]
    rms = math.sqrt(sum(r * r for r in resid) / len(resid)) if resid else 0.0
    return Surrogate(base, effects, slope, rms)


def best_composition(surrogate: Surrogate, objective: Objective, reference_ppl: float) -> tuple[str, ...]:
    """The composition with the lowest predicted perplexity whose predicted decode stays within the cap."""
    assert surrogate.base.decode is not None
    floor = surrogate.base.decode * (1.0 - objective.decode_cap)
    classes = tuple(surrogate.effects)
    best: tuple[float, tuple[str, ...]] | None = None
    for k in range(len(classes) + 1):
        for combo in itertools.combinations(classes, k):
            p = surrogate.predict(combo)
            if p["decode"] < floor:
                continue
            if objective.size_budget_mib is not None and p["size_mib"] > objective.size_budget_mib:
                continue
            if best is None or p["ppl"] < best[0]:
                best = (p["ppl"], combo)
    return best[1] if best else ()


def structured(evaluator: Evaluator, objective: Objective, say: Progress, *,
               classes: tuple[str, ...] = CLASSES, base: str = "Q4_0", verify: int = 2) -> SearchResult:
    """`verify`: how many trials may follow the screen: the surrogate's pick, and, when that pick lands within
    noise of the decode cap (or over it), the same mix without its heaviest class, so a choice made at the
    boundary rests on two measurements rather than one."""
    total = 2 + len(classes) + verify
    say(f"search: the base, the Q8_0 source as the decode anchor, {len(classes)} single-class trials, then up to "
        f"{verify} verification; {total} trials")
    base_trial = evaluator.measure(Candidate.plain(base))
    if base_trial.error or base_trial.ppl is None or base_trial.decode is None:
        raise RuntimeError(f"the base {base} could not be measured: {base_trial.error}")
    say(f"trial 2/{total}: the Q8_0 source itself (no build; pins how decode falls with bytes)")
    anchor = evaluator.measure(Candidate.plain("Q8_0"))
    singles: dict[str, Trial] = {}
    for n, c in enumerate(classes, 3):
        say(f"trial {n}/{total}: {c} alone at Q8_0")
        t = evaluator.measure(Candidate.upgrade([c], base=base))
        if t.error is None:
            singles[c] = t
    surrogate = fit_surrogate(base_trial, singles, anchors=[anchor] if anchor.error is None else None)
    pick = best_composition(surrogate, objective, evaluator.reference())
    forecast = surrogate.predict(pick) if pick else {}
    predicted = {"classes": list(pick), **{k: round(v, 3) for k, v in forecast.items()}} if pick else {}
    trials = [base_trial, anchor, *singles.values()]
    if pick and len(pick) > 1 and verify > 0:
        n = 3 + len(classes)
        say(f"trial {n}/{total}: the surrogate's pick, {', '.join(pick)} at Q8_0 "
            f"(predicted ratio {forecast['ppl'] / evaluator.reference():.3f}, decode {forecast['decode']:.1f})")
        verified = evaluator.measure(Candidate.upgrade(list(pick), base=base))
        trials.append(verified)
        floor = base_trial.decode * (1.0 - objective.decode_cap)
        near_cap = (verified.error is not None or verified.decode is None
                    or verified.decode < floor * (1.0 + objective.decode_noise))
        if near_cap and verify > 1 and len(pick) > 2:
            heaviest = max(pick, key=lambda c: surrogate.effects[c][1])
            neighbour = tuple(c for c in pick if c != heaviest)
            say(f"trial {n + 1}/{total}: that pick sits at the decode cap; also without {heaviest}")
            trials.append(evaluator.measure(Candidate.upgrade(list(neighbour), base=base)))
            predicted["neighbour"] = list(neighbour)
    chosen = objective.choose(trials, base_trial)
    return SearchResult(base_trial, trials, chosen, "structured", evaluator.reference(), surrogate, predicted)


def listed(evaluator: Evaluator, objective: Objective, say: Progress, candidates: list[Candidate],
           base: str = "Q4_0") -> SearchResult:
    base_trial = evaluator.measure(Candidate.plain(base))
    trials = [base_trial]
    for n, cand in enumerate(candidates, 2):
        say(f"trial {n}/{len(candidates) + 1}: {cand.label}")
        trials.append(evaluator.measure(cand))
    return SearchResult(base_trial, trials, objective.choose(trials, base_trial), "listed", evaluator.reference())
