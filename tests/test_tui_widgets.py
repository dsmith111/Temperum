"""The interface's pieces on their own: the load monitor, the gauge bar, the stage checklist, the plan words."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from temperum.core.errors import TemperumError
from temperum.tui import app as tui
from temperum.tui.monitor import Monitor, Sample, parse_sample
from temperum.tui.widgets import ResourceGraph, StageList, bar


def test_the_monitor_parses_samples_and_reads_a_powershell_stream() -> None:
    assert parse_sample("7.83 5463 12.5 2\n") == Sample(7.83, 5463.0, 12.5, 2)
    assert parse_sample("garbage") is None and parse_sample("a b c d") is None
    seen: list[Sample] = []

    class Proc:
        stdout = iter(["3 5000 1.0 2\n", "not a sample\n", "50 4000 2.5 2\n"])

        def terminate(self) -> None:
            seen.append(Sample(-1, -1, -1, -1))

    monitor = Monitor(seen.append, popen=lambda *a, **k: Proc(), platform="win32")
    assert monitor.start()
    import time
    for _ in range(100):
        if len(seen) >= 2:
            break
        time.sleep(0.02)
    assert seen[:2] == [Sample(3.0, 5000.0, 1.0, 2), Sample(50.0, 4000.0, 2.5, 2)]
    monitor.stop()
    assert seen[-1].cpu_percent == -1  # terminated
    assert not Monitor(seen.append, platform="linux").start()  # no readings elsewhere


def test_bar_and_stage_list() -> None:
    assert bar(0.0, 4) == "░░░░" and bar(0.5, 4) == "██░░" and bar(2.0, 4) == "████" and bar(-1, 4) == "░░░░"
    stages = StageList(["download", "convert", "verify", "install"])
    stages.reach("convert")
    assert stages.states == {"download": "done", "convert": "running", "verify": "pending", "install": "pending"}
    assert stages.running == "convert"
    stages.finish(ok=True)
    assert stages.states == {"download": "done", "convert": "done", "verify": "skipped", "install": "skipped"}
    stages = StageList(["a", "b"])
    stages.reach("b")
    stages.finish(ok=False)
    assert stages.states == {"a": "done", "b": "failed"}


def _plan(quant: str, gguf_gb: float, params: float = 1.5) -> Any:
    return SimpleNamespace(quant=quant, estimates={"gguf_gb": gguf_gb, "ram_gb": 3.0, "minutes": 6,
                                                   "params_billions": params, "download_gb": 3.1, "disk_gb": 9.0})


def test_the_recommendation_and_its_words() -> None:
    small = {"Q4_0": _plan("Q4_0", 0.3, 0.49), "Q8_0": _plan("Q8_0", 0.6, 0.49), "Q4_K_M": _plan("Q4_K_M", 0.35, 0.49)}
    assert tui.recommend(small) == ("Q8_0", "a model this small loses noticeable quality at 4 bits, and at 8 bits it is "
                                            "still small and fast")
    mid = {"Q4_0": _plan("Q4_0", 0.9), "Q8_0": _plan("Q8_0", 1.7)}
    quant, why = tui.recommend(mid)
    assert quant == "Q4_0" and why.startswith("at this size 4 bits keeps the quality") and "window" not in why
    big = {"Q4_0": _plan("Q4_0", 4.4, 7.6), "Q8_0": _plan("Q8_0", 7.6, 7.6)}
    quant, why = tui.recommend(big)
    assert quant == "Q4_0" and why.endswith("; Q8_0 would not fit in the NPU's memory window")
    failed = {"Q4_0": TemperumError("plan", "BAD", "no"), "Q8_0": _plan("Q8_0", 1.0, 0.5)}
    assert tui.recommend(failed)[0] == "Q8_0"
    assert tui.recommend({}) == ("Q4_0", "no quantization could be planned")
    label = tui.quant_label("Q8_0", big["Q8_0"], "Q4_0")
    assert label == "Q8_0  best quality  ·  7.6 GB  ·  ~6 min  ·  too big for the NPU: runs split, slowly"
    assert tui.quant_label("Q4_0", big["Q4_0"], "Q4_0").endswith("·  whole on the NPU  ★ recommended")
    assert tui.quant_label("Q4_K_M", failed["Q4_0"], "Q8_0") == "Q4_K_M: not possible here (no)"


def test_the_npu_row_follows_the_pipelines_placement_lines() -> None:
    assert tui.npu_share("on the NPU: 24/24 layers, 400 MiB of weights") == (1.0, "24 of 24 layers (400 MiB)")
    assert tui.npu_share("on the NPU: 12/36 layers, 900.5 MiB of weights") == (1 / 3, "12 of 36 layers (900.5 MiB)")
    assert tui.npu_share("on the NPU: no layers (the server put no weights on an HTP device)") == (
        0.0, "no layers: the server put nothing on the NPU")
    assert tui.npu_share("installed as m:npu; the fork runs it whole on the NPU (126 MiB): ollama run m:npu") == (
        1.0, "the whole model (126 MiB)")
    assert tui.npu_share("the fork runs it 1887 of 3762 MiB on the NPU") == (1887 / 3762, "1887 of 3762 MiB, the rest "
                                                                                          "on the CPU")
    assert tui.npu_share("installed as m; the fork runs it on the CPU: ollama run m") == (0.0, "nothing: the model "
                                                                                                "runs on the CPU")
    assert tui.npu_share("converting to GGUF") is None
    # ten segments: purple for the share on the NPU (never none when any is), grey for the rest
    purple, grey = ResourceGraph.PURPLE, ResourceGraph.GRAY
    assert ResourceGraph.segments(None).count(purple) == 0 and ResourceGraph.segments(None).count(grey) == 10
    assert ResourceGraph.segments(1 / 3).count(purple) == 3 and ResourceGraph.segments(1 / 3).count(grey) == 7
    assert ResourceGraph.segments(0.01).count(purple) == 1 and ResourceGraph.segments(1.0).count(grey) == 0
