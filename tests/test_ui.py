from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from temperum import ui


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=100)


def test_stage_of_maps_the_pipelines_messages() -> None:
    assert ui.stage_of("fetching org/model") == 1
    assert ui.stage_of("converting to GGUF (Q8_0) with llama.cpp's converter") == 2
    assert ui.stage_of("quantizing to Q4_0 with the runtime's llama-quantize") == 2
    assert ui.stage_of("assembling the bundle") == 3
    assert ui.stage_of("prompt 3/6") == 4
    assert ui.stage_of("measuring perplexity (the bundle on the NPU vs the Q8_0 conversion on the CPU)") == 4
    assert ui.stage_of("installing into the fork as x:npu") == 5
    assert ui.stage_of("llama_model_quantize_impl: quant size = 1820 MiB") is None


def test_stage_progress_advances_by_stage_and_never_claims_completion() -> None:
    now = [0.0]
    stages = ui.StageProgress(_console(), estimate_minutes=10, clock=lambda: now[0])
    with stages:
        assert stages.completed() == 0.0
        stages.say("fetching org/model")
        now[0] = 1.0
        assert 0.06 <= stages.completed() < 0.24
        now[0] = 10_000.0  # a stage that overruns its share stops short of the next one
        assert stages.completed() < 0.24
        stages.say("converting to GGUF")
        assert stages.completed() >= 0.24
        stages.say("fetching again")  # messages never move the bar backwards
        assert stages.stage == 2
        stages.say("prompt 1/6")
        now[0] = 10_001.0
        assert 0.52 <= stages.completed() < 0.94
        assert "longer than the estimate" in stages.eta_text()
    now[0] = 30.0
    assert "about 10 min left" in ui.StageProgress(_console(), 10, clock=lambda: now[0]).eta_text()


def test_download_stage_follows_the_snapshot_size(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"x" * 500)
    probe = ui.snapshot_probe(tmp_path, 1000)
    assert probe() == (500, 1000)
    assert ui.snapshot_probe(tmp_path, 0)() is None
    stages = ui.StageProgress(_console(), 10, bytes_probe=probe, clock=lambda: 0.0)
    with stages:
        stages.say("fetching org/model")
        assert abs(stages.completed() - (0.06 + 0.18 * 0.5)) < 1e-6


def test_ready_panel_and_validation_table_render_the_facts() -> None:
    console = _console()
    console.print(ui.ready_panel({
        "bundle": "C:/x/bundle", "verdict": "warning", "reasons": ["quality not checked"],
        "npu_placement": {"layers_offloaded": 29, "layers_total": 29, "htp_weight_mib": 1841},
        "perplexity_ratio": 1.06, "minutes": 5.0,
        "speed": {"median_tokens_per_s": 17.2, "long_prompt_tokens_per_s": 706.0, "long_prompt_tokens": 1749}}))
    text = console.file.getvalue()  # type: ignore[attr-defined]
    assert "Model ready, with warnings" in text and "29/29 layers" in text and "1.06x" in text and "706" in text
    console = _console()
    console.print(ui.validation_table({
        "placement": {"where": "npu", "npu_mib": 2400}, "load_s": 3.1,
        "chat": {"answers": ["Paris.", "Berlin."], "correct": 2, "of": 2},
        "speed": {"decode_tokens_per_s": 21.1, "prompt_tokens_per_s": 379.0, "tokens": 120},
        "cpu": {"cores": 0.12, "processes": 2}}))
    text = console.file.getvalue()  # type: ignore[attr-defined]
    assert "whole on the NPU" in text and "2/2" in text and "21.1" in text and "0.12 cores" in text
    console = _console()
    console.print(ui.validation_table({"placement": {"where": "cpu"}, "cpu": {"skipped": "no fork processes"}}))
    text = console.file.getvalue()  # type: ignore[attr-defined]
    assert "on the CPU" in text and "not measured: no fork processes" in text
