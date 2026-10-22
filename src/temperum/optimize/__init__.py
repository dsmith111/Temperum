"""Temperum as an optimizer for models on the NPU (CONVERTER_PLAN.md, D22).

A search over a configuration space, every candidate measured on this machine's NPU, chosen by an
objective, written into the bundle with its evidence. The first space is the per-tensor quantization mix.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from temperum.gguf import LlamaPayload
from temperum.optimize.evaluate import Evaluator, Trial
from temperum.optimize.objective import Objective
from temperum.optimize.report import lines, markdown, recipe_document, why
from temperum.optimize.space import CLASSES, Candidate
from temperum.optimize.strategies import SearchResult, listed, structured

__all__ = ["CLASSES", "Candidate", "Evaluator", "Objective", "SearchResult", "Trial", "lines",
           "markdown", "recipe_document", "search", "why"]

RECIPES = ("auto",)
PPL_CHUNKS = 16  # the verifier's standard level times four: 8 gave 1.14x where 16 gave 1.09x
SERVER_CONTEXT = 4096
# about 4 min per trial at 1.5B parameters, scaling with the model (quantize, perplexity, server)
TRIAL_MINUTES_PER_BILLION = 2.4
TRIALS = 2 + len(CLASSES) + 2  # the base, the Q8_0 anchor, one trial per class, the pick and (at the cap) its neighbour


def search(payload: LlamaPayload, source: Path, work: Path, heldout: Path, *, say: Callable[[str], None],
           objective: Objective | None = None, strategy: str = "structured", context: int = SERVER_CONTEXT,
           chunks: int = PPL_CHUNKS, candidates: list[Candidate] | None = None) -> SearchResult:
    """Search the quantization mixes of the Q8_0 GGUF at `source` on the NPU; trials cache under `work`."""
    objective = objective or Objective()
    evaluator = Evaluator(payload, source, work, heldout, context=min(context, SERVER_CONTEXT), chunks=chunks,
                          say=say)
    if strategy == "structured":
        return structured(evaluator, objective, say)
    if strategy == "listed":
        return listed(evaluator, objective, say, candidates or [])
    raise ValueError(f"unknown strategy {strategy}")


def estimate_minutes(params_billions: float) -> float:
    """What a structured search adds to a conversion's estimate."""
    return round(TRIALS * max(1.0, TRIAL_MINUTES_PER_BILLION * params_billions), 0)


def document(result: SearchResult, evaluator_reference: float, objective: Objective, *, chunks: int,
             host: dict[str, Any], llama_cpp: str | None, source: dict[str, Any]) -> dict[str, Any]:
    return recipe_document(result, objective, reference_ppl=evaluator_reference, chunks=chunks, host=host,
                           llama_cpp=llama_cpp, source=source)
