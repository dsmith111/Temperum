"""The recipe as a file and as lines for people: every trial with its measurements, the choice and why."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from temperum.optimize.evaluate import DECODE_REQUESTS, DECODE_TOKENS, PPL_CONTEXT, Trial
from temperum.optimize.objective import Objective
from temperum.optimize.strategies import SearchResult

SCHEMA = 1


def recipe_document(result: SearchResult, objective: Objective, *, reference_ppl: float, chunks: int,
                    host: dict[str, Any], llama_cpp: str | None, source: dict[str, Any]) -> dict[str, Any]:
    chosen = result.chosen
    return {
        "schema_version": SCHEMA,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": result.strategy,
        "objective": objective.to_dict(),
        "measurement": {"perplexity": f"llama-perplexity on the NPU, {chunks} chunks of {PPL_CONTEXT} tokens of "
                                      f"wikitext-2 test rows, over the Q8_0 conversion on the CPU "
                                      f"(reference {reference_ppl})",
                        "speed": f"llama-server as the fork runs it (flash attention on, no polling), median of "
                                 f"{DECODE_REQUESTS} requests of {DECODE_TOKENS} new tokens after a ~540-token prompt"},
        "host": host,
        "llama_cpp": llama_cpp,
        "source": source,
        "base": result.base.to_dict(),
        "trials": [t.to_dict() for t in result.trials],
        "surrogate": result.surrogate.to_dict() if result.surrogate else None,
        "predicted": result.predicted,
        "chosen": {**chosen.to_dict(), "label": chosen.candidate.label, "why": why(result, objective)},
    }


def why(result: SearchResult, objective: Objective) -> str:
    chosen, base = result.chosen, result.base
    if chosen is base:
        return (f"nothing measured beat plain {base.candidate.base} by more than the noise "
                f"({objective.ratio_noise} in the ratio) within {objective.decode_cap:.0%} of its decode")
    assert chosen.ratio is not None and base.ratio is not None
    cost = ((base.decode - chosen.decode) / base.decode) if base.decode and chosen.decode else 0.0
    speed = f"{cost:.0%} slower" if cost > 0 else f"{-cost:.0%} faster"
    return (f"perplexity ratio {chosen.ratio} against {base.ratio} for plain {base.candidate.base}, "
            f"decode {chosen.decode} against {base.decode} tok/s ({speed}), {chosen.size_mib:.0f} against "
            f"{base.size_mib:.0f} MiB; within the {objective.decode_cap:.0%} decode cap")


def trial_rows(trials: list[Trial]) -> list[tuple[str, str, str, str, str, str]]:
    rows = []
    for t in sorted(trials, key=lambda t: (t.ratio is None, t.ratio or 0.0)):
        rows.append((t.candidate.label, f"{t.size_mib:.0f}", _f(t.ratio, 3), _f(t.decode, 1), _f(t.prefill, 0),
                     t.layers or ("failed" if t.error else "")))
    return rows


def lines(result: SearchResult, objective: Objective) -> list[str]:
    """Plain lines for a log or the ready panel."""
    out = [f"recipe: {result.chosen.candidate.label} ({why(result, objective)})"]
    for label, size, ratio, decode, prefill, layers in trial_rows(result.trials):
        out.append(f"  {label}: {size} MiB, ratio {ratio}, decode {decode} tok/s, prefill {prefill}, layers {layers}")
    return out


def _f(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"
