"""What runs where: ggml's scheduler assignments read back from llama-server's debug log.

With `GGML_SCHED_DEBUG=2` and debug logging, llama-server prints every graph it schedules as splits per backend
and one line per node:

    ## SPLIT #1: HTP0 # 9 inputs: [inp_mixed_embd (   6K)] ...
    node #  6 (   MUL_MAT):               Qcur-0 (   6K) [ HTP0         ] use=1,c=1:   blk.0.attn_q.weight ( 2M) [ HTP0 ] ...

llama-server prefixes every fragment with a timestamp and a level letter; this module strips those, groups nodes
into graphs, and summarises one graph: how many nodes each backend ran, how many splits (each boundary is a copy
between the CPU and the NPU), and which operations fell back to the CPU with the tensors they touched. That is the
explanation behind a model's cores-busy figure and the diagnostic the runtime space will need.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Any

_PREFIX = re.compile(r"\d+\.\d+\.\d+\.\d+ [A-Z] ")
_SPLIT = re.compile(r"## SPLIT #(\d+): (\S+) # (\d+) inputs")
_NODE = re.compile(r"node #\s*(\d+) \(\s*([A-Z_0-9]+)\):\s*(\S+) \(\s*([0-9.]+[KMG]?)\) \[\s*(\S+)\s*\]")
_SOURCE = re.compile(r"\s(\S+) \(\s*[0-9.]+[KMG]?\) \[\s*(\S+)\s*\]")


@dataclass
class Node:
    index: int
    op: str
    name: str
    backend: str
    sources: list[str] = field(default_factory=list)  # tensor names the node reads (weights among them)


@dataclass
class Split:
    index: int
    backend: str
    inputs: int
    nodes: list[Node] = field(default_factory=list)


@dataclass
class Graph:
    splits: list[Split] = field(default_factory=list)

    @property
    def nodes(self) -> list[Node]:
        return [n for s in self.splits for n in s.nodes]


def clean(text: str) -> str:
    """Strip llama-server's per-fragment prefixes so the scheduler's own lines read as printed."""
    return _PREFIX.sub("", text)


def parse(text: str) -> list[Graph]:
    """Every scheduled graph in the log, in order (a prompt graph, then one per generated token)."""
    graphs: list[Graph] = []
    current: Graph | None = None
    split: Split | None = None
    for line in clean(text).splitlines():
        m = _SPLIT.search(line)
        if m:
            index = int(m.group(1))
            if index == 0 or current is None:
                current = Graph()
                graphs.append(current)
            split = Split(index, m.group(2), int(m.group(3)))
            current.splits.append(split)
            continue
        m = _NODE.search(line)
        if m and split is not None:
            node = Node(int(m.group(1)), m.group(2), m.group(3), m.group(5))
            rest = line[m.end():]
            node.sources = [name for name, _backend in _SOURCE.findall(rest) if not name.startswith(("HTP", "CPU#"))]
            split.nodes.append(node)
    return graphs


def summarise(graph: Graph, npu_prefix: str = "HTP") -> dict[str, Any]:
    """One graph as a report: nodes per backend, splits, and what fell back to the CPU."""
    per_backend: collections.Counter[str] = collections.Counter(n.backend for n in graph.nodes)
    fallbacks: dict[str, dict[str, Any]] = {}
    for n in graph.nodes:
        if n.backend.startswith(npu_prefix):
            continue
        entry = fallbacks.setdefault(n.op, {"count": 0, "tensors": []})
        entry["count"] += 1
        for src in n.sources:
            if src.endswith(".weight") and src not in entry["tensors"] and len(entry["tensors"]) < 4:
                entry["tensors"].append(src)
    total = len(graph.nodes)
    on_npu = sum(c for b, c in per_backend.items() if b.startswith(npu_prefix))
    return {"nodes": total, "on_npu": on_npu, "per_backend": dict(per_backend), "splits": len(graph.splits),
            "copies": sum(s.inputs for s in graph.splits if s.backend.startswith(npu_prefix)),
            "fallbacks": fallbacks}


def describe(summary: dict[str, Any]) -> str:
    """One line for people: `434 of 436 ops on the NPU in 2 splits; on the CPU: GET_ROWS x2 (token_embd.weight)`."""
    parts = [f"{summary['on_npu']} of {summary['nodes']} ops on the NPU in {summary['splits']} split"
             + ("" if summary["splits"] == 1 else "s")]
    if summary["fallbacks"]:
        items = []
        for op, entry in sorted(summary["fallbacks"].items(), key=lambda kv: -kv[1]["count"]):
            tensors = f" ({', '.join(entry['tensors'])})" if entry["tensors"] else ""
            items.append(f"{op} x{entry['count']}{tensors}")
        parts.append("on the CPU: " + ", ".join(items))
    return "; ".join(parts)


def decode_graph(graphs: list[Graph]) -> Graph | None:
    """The steady-state graph: the last one printed (a generated token), or the only one."""
    return graphs[-1] if graphs else None
