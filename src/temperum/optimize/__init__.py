"""The recipe search: the quantization mix, and the CPU threads beside the NPU, that run best here.

A fixed recipe does not hold across models (Qwen2.5-1.5B gains from Q8_0 attention, Llama-3.2-3B does
not), so the conversion measures candidates on this machine's NPU, chooses by an objective and writes the
choice into the bundle with its evidence. It is one of the things Temperum does so a model runs well on
the NPU. How it searches is an implementation detail: a screen and an additive surrogate today, never a
grid of everything, no tuning library.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from temperum.gguf import LlamaPayload
from temperum.optimize.evaluate import Evaluator, RuntimeTrial, Trial
from temperum.optimize.objective import Objective
from temperum.optimize.report import lines, markdown, recipe_document, why
from temperum.optimize.space import CLASSES, Candidate
from temperum.optimize.strategies import RuntimeResult, SearchResult, listed, structured, threads_search

__all__ = ["CLASSES", "Candidate", "Evaluator", "Objective", "RuntimeResult", "RuntimeTrial", "SearchResult",
           "Trial", "lines", "markdown", "recipe_document", "runtime_search", "search", "threads_search", "why"]

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


def runtime_search(payload: LlamaPayload, model: Path, work: Path, heldout: Path, *, say: Callable[[str], None],
                   context: int = SERVER_CONTEXT) -> RuntimeResult:
    """The runtime space for a built GGUF: the CPU threads llama-server runs beside the NPU. About half a
    minute a trial, no quantizing; the winner goes into the bundle's Modelfile as `num_thread`."""
    evaluator = Evaluator(payload, model, work, heldout, context=min(context, SERVER_CONTEXT), chunks=PPL_CHUNKS,
                          say=say)
    return threads_search(evaluator, model, say)


RUNTIME_MINUTES = 2.0  # three thread settings through llama-server


def estimate_minutes(params_billions: float) -> float:
    """What a structured search plus the runtime pass adds to a conversion's estimate."""
    return round(TRIALS * max(1.0, TRIAL_MINUTES_PER_BILLION * params_billions) + RUNTIME_MINUTES, 0)


def document(result: SearchResult, evaluator_reference: float, objective: Objective, *, chunks: int,
             host: dict[str, Any], llama_cpp: str | None, source: dict[str, Any],
             runtime: RuntimeResult | None = None) -> dict[str, Any]:
    return recipe_document(result, objective, reference_ppl=evaluator_reference, chunks=chunks, host=host,
                           llama_cpp=llama_cpp, source=source, runtime=runtime)
