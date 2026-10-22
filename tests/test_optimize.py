"""The optimizer (D22): the space, the objective, the additive surrogate and the structured search, with a
synthetic model standing in for the NPU; the pipeline's hook through the planner."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
from test_gguf import X_ELITE, _model, _payload, _version

from temperum import gguf, optimize
from temperum.core.errors import TemperumError
from temperum.optimize import strategies
from temperum.optimize.evaluate import Trial
from temperum.optimize.objective import Objective
from temperum.optimize.space import CLASSES, Candidate
from temperum.profiles import get_target

# a synthetic model: each class alone at Q8_0 lowers log-perplexity by `gain` and adds `mib`; decode falls
# with the extra bytes at a fixed rate, with a little deterministic "noise" on the ratio
EFFECTS: dict[str, tuple[float, float]] = {
    "attn_q": (-0.003, 31.5), "attn_k": (-0.003, 5.3), "attn_v": (-0.025, 5.3), "attn_output": (-0.016, 31.5),
    "ffn_gate": (-0.015, 184.0), "ffn_up": (-0.027, 184.0), "ffn_down": (-0.024, 184.0),
}
BASE_PPL, BASE_MIB, BASE_DECODE, REFERENCE, SLOPE = 13.36, 891.6, 32.0, 11.62, 0.0088


class FakeEvaluator:
    """Measures candidates against the synthetic model; counts what it was asked to build."""

    def __init__(self, work: Path, noise: float = 0.0) -> None:
        self.work, self.noise = work, noise
        self.measured: list[str] = []

    def reference(self) -> float:
        return REFERENCE

    def measure(self, candidate: Candidate) -> Trial:
        self.measured.append(candidate.name)
        if candidate.base == "Q8_0":  # the anchor: the source itself, 678 MiB heavier, 1.035x on the NPU
            return Trial(candidate, self.work / "src.gguf", BASE_MIB + 678.0, round(REFERENCE * 1.035, 4), 1.035,
                         round(BASE_DECODE - SLOPE * 678.0, 1), 1300.0, "29/29", True, 1.0)
        dlog = sum(EFFECTS[c][0] for c in candidate.classes)
        extra = sum(EFFECTS[c][1] for c in candidate.classes)
        ppl = BASE_PPL * math.exp(dlog) * (1 + self.noise * (len(candidate.classes) % 2))
        decode = BASE_DECODE - SLOPE * extra
        return Trial(candidate, self.work / f"model-{candidate.name}.gguf", BASE_MIB + extra, round(ppl, 4),
                     round(ppl / REFERENCE, 4), round(decode, 1), 1300.0, "29/29", True, 1.0)


def test_candidates_name_themselves_and_their_quantizer_flags() -> None:
    plain = Candidate.plain()
    mix = Candidate.upgrade(["ffn_up", "attn_v"])
    assert plain.name == "q4_0" and plain.label == "Q4_0" and plain.quantize_args() == []
    assert mix.name == "q4_0+attn_v.q8_0+ffn_up.q8_0"  # in block order, whatever order they were given
    assert mix.label == "Q4_0, attn_v and ffn_up at Q8_0"
    assert mix.quantize_args() == ["--tensor-type", "attn_v=q8_0", "--tensor-type", "ffn_up=q8_0"]
    assert Trial.from_dict(json.loads(json.dumps(FakeEvaluator(Path(".")).measure(mix).to_dict()))).candidate == mix


def _trial(name: str, ratio: float, decode: float, size: float, whole: bool = True) -> Trial:
    classes = [c for c in CLASSES if c in name]
    return Trial(Candidate.upgrade(classes) if classes else Candidate.plain(), Path(name), size, ratio * REFERENCE,
                 ratio, decode, 1300.0, "29/29" if whole else "20/29", whole, 1.0)


def test_the_objective_keeps_the_rule_and_treats_noise_as_ties() -> None:
    objective = Objective(decode_cap=0.10, ratio_noise=0.02, decode_noise=0.05)
    base = _trial("q4_0", 1.150, 32.0, 892)
    better = _trial("attn_v", 1.097, 33.6, 965)
    too_slow = _trial("ffn_down", 1.020, 27.0, 1149)      # 16% slower: over the cap
    split = _trial("ffn_up", 1.010, 33.0, 1075, whole=False)
    assert objective.choose([base, better, too_slow, split], base) is better
    # a tie on quality goes to the smaller file, then the faster decode
    same_quality_bigger = _trial("attn_output", 1.090, 33.0, 1000)
    assert objective.choose([base, better, same_quality_bigger], base) is better
    assert objective.choose([base], base) is base
    # 10% cap with 5% noise: 28.8 would be the floor, 27.5 still passes the noise allowance, 27.0 does not
    assert objective.admissible(_trial("attn_q", 1.1, 27.5, 900), base)
    assert not objective.admissible(_trial("attn_q", 1.1, 27.0, 900), base)


def test_the_surrogate_adds_single_effects_and_fits_decode_to_bytes(tmp_path: Path) -> None:
    ev = FakeEvaluator(tmp_path)
    base = ev.measure(Candidate.plain())
    singles = {c: ev.measure(Candidate.upgrade([c])) for c in CLASSES}
    surrogate = strategies.fit_surrogate(base, singles)
    assert surrogate.slope == pytest.approx(SLOPE, rel=0.05) and surrogate.rms < 0.1  # decode rounds to 0.1
    attention = ("attn_q", "attn_k", "attn_v", "attn_output")
    predicted = surrogate.predict(attention)
    measured = ev.measure(Candidate.upgrade(list(attention)))
    assert predicted["ppl"] == pytest.approx(measured.ppl, rel=1e-3)
    assert predicted["decode"] == pytest.approx(measured.decode, abs=0.1)
    pick = strategies.best_composition(surrogate, Objective(decode_cap=0.10), REFERENCE)
    # attention plus ffn_up fits under the cap (257 MiB -> 29.7 tok/s); adding ffn_down would not
    assert set(pick) == {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_up"}


def test_structured_search_screens_once_per_class_then_verifies_its_pick(tmp_path: Path) -> None:
    ev = FakeEvaluator(tmp_path)
    said: list[str] = []
    result = strategies.structured(ev, Objective(), said.append)  # type: ignore[arg-type]
    # the base, the Q8_0 anchor, the singles, the pick (29.7 tok/s: within 5% of the 28.8 floor) and its neighbour
    assert len(ev.measured) == 2 + len(CLASSES) + 2
    assert result.chosen.candidate.classes == ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_up")
    assert result.chosen in result.trials and result.surrogate is not None
    assert result.predicted["classes"] == list(result.chosen.candidate.classes)
    assert result.predicted["neighbour"] == ["attn_q", "attn_k", "attn_v", "attn_output"]
    assert any("verification" in s for s in said)
    doc = optimize.document(result, REFERENCE, Objective(), chunks=16, host={"system": "Windows"},
                            llama_cpp="e467e5b", source={"repo": "Qwen/Qwen2.5-1.5B-Instruct"})
    assert doc["chosen"]["label"] == "Q4_0, attn_q, attn_k, attn_v, attn_output and ffn_up at Q8_0"
    assert "within the 10% decode cap" in doc["chosen"]["why"] and len(doc["trials"]) == 11
    json.dumps(doc)  # serialisable as written
    lines = optimize.lines(result, Objective())
    assert lines[0].startswith("recipe: Q4_0, attn_q") and len(lines) == 12


def test_a_search_that_finds_nothing_keeps_plain_q4_0(tmp_path: Path) -> None:
    class Flat(FakeEvaluator):
        def measure(self, candidate: Candidate) -> Trial:
            t = super().measure(candidate)
            return Trial(t.candidate, t.path, t.size_mib, BASE_PPL, round(BASE_PPL / REFERENCE, 4), t.decode,
                         t.prefill, t.layers, t.whole, t.seconds)

    result = strategies.structured(Flat(tmp_path), Objective(), lambda _m: None)  # type: ignore[arg-type]
    assert result.chosen is result.base
    assert optimize.why(result, Objective()).startswith("nothing measured beat plain Q4_0")


def test_a_failed_trial_is_recorded_and_skipped(tmp_path: Path) -> None:
    class Flaky(FakeEvaluator):
        def measure(self, candidate: Candidate) -> Trial:
            if candidate.classes == ("ffn_up",):
                return Trial(candidate, self.work / "x", 0.0, None, None, None, None, None, False, 0.1, "boom")
            return super().measure(candidate)

    result = strategies.structured(Flaky(tmp_path), Objective(), lambda _m: None)  # type: ignore[arg-type]
    assert result.surrogate is not None and "ffn_up" not in result.surrogate.effects
    assert result.chosen.candidate.classes == ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_down")


def test_the_planner_takes_a_recipe_and_adds_its_time(tmp_path: Path) -> None:
    exe = _payload(tmp_path)
    run = _version("version: 0.6.0-dev (build 1, commit e467e5b)")
    target = get_target("gguf-hexagon")
    plain = gguf.plan(_model(), X_ELITE, target, ollama_exe=exe, run=run, out=tmp_path / "a")
    searched = gguf.plan(_model(), X_ELITE, target, ollama_exe=exe, run=run, out=tmp_path / "b", recipe="auto")
    assert plain.recipe is None and searched.recipe == "auto" and searched.to_dict()["recipe"] == "auto"
    assert searched.estimates["minutes"] > plain.estimates["minutes"]
    with pytest.raises(TemperumError, match="Unknown recipe grid"):
        gguf.plan(_model(), X_ELITE, target, ollama_exe=exe, run=run, out=tmp_path / "c", recipe="grid")
    with pytest.raises(TemperumError, match="Q4_0 base"):
        gguf.plan(_model(), X_ELITE, target, ollama_exe=exe, run=run, out=tmp_path / "d", recipe="auto",
                  quant="Q8_0")


def test_quantize_passes_recipe_overrides_before_the_paths(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class Proc:
        returncode = 0
        stdout = iter(["llama_model_quantize_impl: quant size = 965.10 MB\n"])

        def wait(self) -> int:
            return 0

    def popen(cmd: list[str], **_kwargs: Any) -> Proc:
        seen["cmd"] = cmd
        (tmp_path / "out.gguf.partial").write_bytes(b"gguf")
        return Proc()

    payload = gguf.LlamaPayload(tmp_path / "lib")
    gguf.quantize(payload, tmp_path / "src.gguf", tmp_path / "out.gguf", "Q4_0", tmp_path / "log", lambda _m: None,
                  popen=popen, overrides=["--tensor-type", "attn_v=q8_0"])
    assert seen["cmd"][1:4] == ["--allow-requantize", "--tensor-type", "attn_v=q8_0"]
    assert seen["cmd"][-1] == "Q4_0" and (tmp_path / "out.gguf").is_file()


@pytest.mark.parametrize("penalty,keeps_pick", [
    (1.6, True),   # 28.1 tok/s: under the 28.8 floor but inside its 5% noise, and clearly better quality: kept
    (3.0, False),  # 26.7 tok/s: out, so the neighbour measured beside it is the choice
])
def test_a_pick_at_the_decode_cap_gets_its_neighbour_measured_too(tmp_path: Path, penalty: float,
                                                                  keeps_pick: bool) -> None:
    class Steep(FakeEvaluator):
        """Compositions decode worse than the singles predict: the pick lands around the cap's floor."""

        def measure(self, candidate: Candidate) -> Trial:
            t = super().measure(candidate)
            cost = penalty if len(candidate.classes) > 1 else 0.0
            return Trial(t.candidate, t.path, t.size_mib, t.ppl, t.ratio, round((t.decode or 0) - cost, 1),
                         t.prefill, t.layers, t.whole, t.seconds)

    ev = Steep(tmp_path)
    said: list[str] = []
    result = strategies.structured(ev, Objective(), said.append)  # type: ignore[arg-type]
    assert len(ev.measured) == 2 + len(CLASSES) + 2
    assert any("sits at the decode cap" in s for s in said)
    assert result.predicted["neighbour"] == ["attn_q", "attn_k", "attn_v", "attn_output"]  # ffn_up was heaviest
    pick = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_up")
    assert result.chosen.candidate.classes == (pick if keeps_pick else pick[:-1])


def test_the_markdown_recipe_lists_the_trials_and_how_to_reproduce_the_choice(tmp_path: Path) -> None:
    result = strategies.structured(FakeEvaluator(tmp_path), Objective(), lambda _m: None)  # type: ignore[arg-type]
    doc = optimize.document(result, REFERENCE, Objective(), chunks=16, host={}, llama_cpp="e467e5b", source={})
    page = optimize.markdown(doc)
    assert page.startswith("# Quantization recipe, measured on this machine's NPU")
    assert "**Chosen: Q4_0, attn_q, attn_k, attn_v, attn_output and ffn_up at Q8_0.**" in page
    assert "`--allow-requantize --tensor-type attn_q=q8_0" in page and "<Q8_0 source> <out> Q4_0`" in page
    assert page.count("\n| Q4_0") == 10 and page.count("\n| Q8_0") == 1 and "**(chosen)**" in page


class ThreadsFake(FakeEvaluator):
    """A model whose per-token CPU split makes it pay for every thread (GPT-2's shape): one thread is 3x
    faster and frees the cores; `flat=True` models one that does not care."""

    def __init__(self, work: Path, flat: bool = False) -> None:
        super().__init__(work)
        self.flat = flat
        self.runtime_measured: list[int | None] = []

    def measure_runtime(self, model: Path, threads: int | None):  # type: ignore[no-untyped-def]
        from temperum.optimize.evaluate import RuntimeTrial

        self.runtime_measured.append(threads)
        if self.flat:
            return RuntimeTrial(threads, 64.4 + (0.3 if threads == 2 else 0.0), 2250.0, 0.08, 1.0)
        table = {None: (38.9, 2.03), 1: (115.8, 0.10), 2: (95.9, 0.22)}
        decode, cores = table[threads]
        return RuntimeTrial(threads, decode, 4100.0, cores, 1.0)


def test_threads_search_keeps_one_thread_when_it_is_faster_and_the_default_when_nothing_is(tmp_path: Path) -> None:
    ev = ThreadsFake(tmp_path)
    said: list[str] = []
    result = strategies.threads_search(ev, tmp_path / "model.gguf", said.append)  # type: ignore[arg-type]
    assert ev.runtime_measured == [None, 1, 2]
    assert result.chosen.threads == 1 and result.params == {"num_thread": 1}
    assert result.why.startswith("1 thread: decode 115.8 against 38.9 tok/s with the default (+198%), 0.1 against 2.03")
    flat = strategies.threads_search(ThreadsFake(tmp_path, flat=True), tmp_path / "m.gguf", lambda _m: None)  # type: ignore[arg-type]
    assert flat.chosen.threads is None and flat.params == {}
    assert flat.why == "default threads: no setting decoded faster beyond noise or freed cores"
    doc = result.to_dict()
    assert doc["chosen"]["params"] == {"num_thread": 1} and len(doc["trials"]) == 3


def test_the_runtime_choice_lands_in_the_modelfile_and_survives_the_template_rewrite(tmp_path: Path) -> None:
    from types import SimpleNamespace

    text = gguf.modelfile(4096, params={"num_thread": 1})
    assert text == "FROM ./model.gguf\nPARAMETER num_ctx 4096\nPARAMETER num_thread 1\n"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "Modelfile").write_text(text, encoding="utf-8")
    (bundle / gguf.MANIFEST).write_text(json.dumps({"files": {}, "modelfile_params": {"num_thread": 1}}),
                                        encoding="utf-8")
    chat = SimpleNamespace(template="{{ .Prompt }}", stops=["<|end|>"], family="raw", variant=None)
    gguf.set_template(bundle, chat, 4096)  # type: ignore[arg-type]
    rewritten = (bundle / "Modelfile").read_text(encoding="utf-8")
    assert "PARAMETER num_thread 1\n" in rewritten and 'PARAMETER stop "<|end|>"' in rewritten


def test_the_recipe_document_and_page_carry_the_runtime_section(tmp_path: Path) -> None:
    ev = ThreadsFake(tmp_path)
    result = strategies.structured(ev, Objective(), lambda _m: None)  # type: ignore[arg-type]
    rt = strategies.threads_search(ev, result.chosen.path, lambda _m: None)  # type: ignore[arg-type]
    doc = optimize.document(result, REFERENCE, Objective(), chunks=16, host={}, llama_cpp="e467e5b", source={},
                            runtime=rt)
    assert doc["runtime"]["chosen"]["params"] == {"num_thread": 1}
    page = optimize.markdown(doc)
    assert "## Runtime: 1 thread" in page and "`PARAMETER num_thread 1`" in page
    assert page.count("| 1 thread **(chosen)** |") == 1 and "| default threads |" in page
    lines = optimize.lines(result, Objective(), rt)
    assert any(line.startswith("runtime: 1 thread (1 thread: decode 115.8") for line in lines)
    assert sum(1 for line in lines if line.startswith("  ") and "cores" in line) == 3


def test_the_search_shows_up_in_the_stage_bar_and_the_result_views() -> None:
    from temperum import flows, ui

    assert ui.stage_of("searching quantization mixes on the NPU (each one built, scored and timed)") == 2
    assert ui.stage_of("trial 3/10: attn_v alone at Q8_0") == 2
    assert ui.stage_of("Q4_0, attn_v at Q8_0: perplexity on the NPU") is None  # within the stage
    summary = {"bundle": "b", "verdict": "passed", "reasons": [], "minutes": 48.0,
               "recipe": {"label": "Q4_0, attn_v and ffn_up at Q8_0", "name": "x", "trials": 10,
                          "why": "perplexity ratio 1.073 against 1.15 for plain Q4_0"}}
    assert any("recipe: Q4_0, attn_v and ffn_up at Q8_0 after 10 trials" in line for line in flows.ready_lines(summary))
    from rich.console import Console

    console = Console(record=True, width=120, file=open(os.devnull, "w", encoding="utf-8"))
    console.print(ui.ready_panel(summary))
    text = console.export_text()
    assert "Recipe" in text and "after 10 trials on the NPU" in text
