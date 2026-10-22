"""What runs where: ggml's scheduler log read back, and verification's short capture of it."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from temperum import gguf_verify, schedlog

# the shape llama-server prints at debug level with GGML_SCHED_DEBUG=2 (timestamps per fragment), two graphs
LOG = """0.00.498.971 D ## SPLIT #0: CPU # 0 inputs0.00.498.971 D
0.00.498.973 D node #  0 (  GET_ROWS):                 embd (   6K) [  CPU         ] use=2,c=1:0.00.498.973 D     token_embd.weight ( 182M) [  CPU         ]0.00.498.974 D            inp_tokens (   0K) [  CPU         ]0.00.498.974
0.00.498.975 D node #  1 (  GET_ROWS):               node_1 (   6K) [  CPU         ] use=1,c=0:0.00.498.975 D     token_embd.weight ( 182M) [  CPU         ]0.00.498.976 D      inp_mixed_tokens (   0K) [  CPU         ]0.00.498.976
## SPLIT #1: HTP0 # 9 inputs0.00.498.976 D : 0.00.498.976 D [inp_mixed_embd (   6K)] 0.00.498.976 D [node_1 (   6K)]
0.00.498.979 D node #  2 (       DUP):               node_2 (   6K) [ HTP0         ] use=1,c=0:0.00.498.979 D  HTP0#inp_mixed_embd# (   6K) [ NULL         ]0.00.498.979 D
0.00.498.984 D node #  6 (   MUL_MAT):               Qcur-0 (   6K) [ HTP0         ] use=1,c=1:0.00.498.985 D   blk.0.attn_q.weight (   2M) [ HTP0         ]0.00.498.985 D           attn_norm-0 (   6K) [ HTP0         ]0.00.498.985
0.00.498.987 D node #  9 (      ROPE):               Qcur-0 (   6K) [ HTP0         ] use=1,c=1:0.00.498.987 D                Qcur-0 (   6K) [ HTP0         ]0.00.498.988 D        HTP0#inp_pos#0 (   0K) [ NULL         ]0.00.498.988
0.00.901.248 D ## SPLIT #0: CPU # 0 inputs0.00.901.248 D
0.00.901.250 D node #  0 (  GET_ROWS):                 embd (   6K) [  CPU         ] use=2,c=1:0.00.901.250 D     token_embd.weight ( 182M) [  CPU         ]0.00.901.251 D
## SPLIT #1: HTP0 # 9 inputs0.00.901.252 D : 0.00.901.252 D [inp_mixed_embd (   6K)]
0.00.901.254 D node #  6 (   MUL_MAT):               Qcur-0 (   6K) [ HTP0         ] use=1,c=1:0.00.901.255 D   blk.0.attn_q.weight (   2M) [ HTP0         ]0.00.901.255 D
0.00.901.256 D node #  7 (      GELU):               ffn-0 (   6K) [  CPU         ] use=1,c=1:0.00.901.256 D   blk.0.ffn_up.weight (   8M) [ HTP0         ]0.00.901.257 D
## SPLIT #2: CPU # 1 inputs0.00.901.258 D : 0.00.901.258 D [ffn-0 (   6K)]
0.00.901.259 D node #  8 (      GELU):               ffn-0 (   6K) [  CPU         ] use=1,c=1:0.00.901.259 D   blk.0.ffn_up.weight (   8M) [ HTP0         ]0.00.901.260 D
"""


def test_parse_strips_the_servers_prefixes_and_groups_nodes_into_graphs() -> None:
    graphs = schedlog.parse(LOG)
    assert len(graphs) == 2
    first = graphs[0]
    assert [s.backend for s in first.splits] == ["CPU", "HTP0"] and first.splits[1].inputs == 9
    assert [n.op for n in first.nodes] == ["GET_ROWS", "GET_ROWS", "DUP", "MUL_MAT", "ROPE"]
    assert first.nodes[3].sources == ["blk.0.attn_q.weight", "attn_norm-0"]  # the NPU's own copies are not sources
    assert first.nodes[0].backend == "CPU" and first.nodes[2].backend == "HTP0"


def test_summary_counts_backends_splits_copies_and_names_the_fallbacks() -> None:
    graphs = schedlog.parse(LOG)
    steady = schedlog.decode_graph(graphs)
    assert steady is graphs[-1]
    s = schedlog.summarise(steady)
    assert s["nodes"] == 4 and s["on_npu"] == 1 and s["per_backend"] == {"CPU": 3, "HTP0": 1}
    assert s["splits"] == 3 and s["copies"] == 9
    assert s["fallbacks"] == {"GET_ROWS": {"count": 1, "tensors": ["token_embd.weight"]},
                              "GELU": {"count": 2, "tensors": ["blk.0.ffn_up.weight"]}}
    assert schedlog.describe(s) == ("1 of 4 ops on the NPU in 3 splits; on the CPU: GELU x2 (blk.0.ffn_up.weight), "
                                    "GET_ROWS x1 (token_embd.weight)")
    whole = schedlog.summarise(graphs[0])
    assert schedlog.describe(whole) == "3 of 5 ops on the NPU in 2 splits; on the CPU: GET_ROWS x2 (token_embd.weight)"
    assert schedlog.decode_graph([]) is None


class FakeServer:
    """Stands in for llama-server: records how it was configured, returns a canned log."""

    text_out = LOG
    made: list[FakeServer] = []

    def __init__(self, payload: Any, model: Path, context: int, log: Path, device: str = "HTP0") -> None:
        self.args = (payload, model, context, log)
        self.extra_env: dict[str, str] = {}
        self.extra_args: list[str] = []
        self.completed: list[tuple[Any, int]] = []
        FakeServer.made.append(self)

    def __enter__(self) -> FakeServer:
        return self

    def __exit__(self, *_exc: object) -> None: ...

    def complete(self, prompt: Any, n_predict: int) -> dict[str, Any]:
        self.completed.append((prompt, n_predict))
        return {"content": " Paris"}

    def text(self) -> str:
        return self.text_out


def test_op_placement_runs_one_short_completion_with_the_scheduler_log(monkeypatch: pytest.MonkeyPatch,
                                                                        tmp_path: Path) -> None:
    monkeypatch.setattr(gguf_verify, "LlamaServer", FakeServer)
    FakeServer.made.clear()
    said: list[str] = []
    ops = gguf_verify.op_placement(object(), tmp_path / "model.gguf", tmp_path, said.append)
    server = FakeServer.made[-1]
    assert server.extra_env == {"GGML_SCHED_DEBUG": "2"} and server.extra_args == ["--log-verbosity", "5"]
    assert server.args[2] == 512 and server.completed == [("The capital of France is", 2)]
    assert ops is not None and ops["on_npu"] == 1 and "GELU" in ops["fallbacks"]
    assert said and "which operations the NPU runs" in said[0]

    class Quiet(FakeServer):
        text_out = "load_tensors: offloaded 29/29 layers to GPU\n"

    monkeypatch.setattr(gguf_verify, "LlamaServer", Quiet)
    assert gguf_verify.op_placement(object(), tmp_path / "model.gguf", tmp_path, lambda _m: None) is None


def test_the_placement_line_carries_the_ops_when_known() -> None:
    placed = {"status": "weights-on-npu", "layers_offloaded": 29, "layers_total": 29, "htp_weight_mib": 898.1}
    assert gguf_verify.placement_line(placed) == "on the NPU: 29/29 layers, 898.1 MiB of weights"
    placed["ops"] = schedlog.summarise(schedlog.parse(LOG)[0])
    assert gguf_verify.placement_line(placed).endswith("; 3 of 5 ops on the NPU in 2 splits; on the CPU: GET_ROWS x2 "
                                                       "(token_embd.weight)")
