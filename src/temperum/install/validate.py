"""Validate an installed model through the fork: where it runs, how fast, and the CPU it keeps busy.

Everything goes through the server's API, as a user's chat would. The CPU figure is the processor time
of the fork's own processes (server and runner, found by their folder) over the wall time of a generation,
which is what the NPU is for: a model generating there leaves the cores to the user.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from temperum.install.ollama_import import _api, _canonical_tag, placement_of

CHAT: tuple[tuple[str, str], ...] = (("What is the capital of France? Answer in one word.", "paris"),
                                     ("And of Germany? Answer in one word.", "berlin"))
LONG_PROMPT = "Explain in about 150 words why the sky is blue."
TOKENS = 120
REQUEST_TIMEOUT = 900.0
Api = Callable[..., dict[str, Any]]
CpuProbe = Callable[[Path | None], "tuple[float, int] | None"]


def cpu_seconds(fork_dir: Path | None, run: Callable[..., Any] = subprocess.run) -> tuple[float, int] | None:
    """Processor seconds used so far by the processes running from the fork's folder, and how many there are."""
    if fork_dir is None or sys.platform != "win32":
        return None
    prefix = str(fork_dir).rstrip("\\") + "\\"
    script = ("$p = @(Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith('"
              + prefix.replace("'", "''") + "') }); "
              "$s = ($p | ForEach-Object { $_.TotalProcessorTime.TotalSeconds } | Measure-Object -Sum).Sum; "
              "[string]::Format([System.Globalization.CultureInfo]::InvariantCulture, '{0} {1}', [double]$s, "
              "$p.Count)")
    try:
        result = run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True,
                     text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    parts = str(result.stdout).split()
    try:
        return float(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


def validate(endpoint: str, tag: str, *, fork_dir: Path | None, say: Callable[[str], None],
             api: Api = _api, cpu: CpuProbe = cpu_seconds) -> dict[str, Any]:
    """Load the model, read its placement, ask two questions, then time a generation while watching the CPU."""
    report: dict[str, Any] = {"tag": tag, "endpoint": endpoint}
    say(f"loading {tag} through the fork")
    started = time.monotonic()
    api(endpoint, "/api/generate", REQUEST_TIMEOUT, {"model": tag, "prompt": "Hi", "stream": False,
                                                     "keep_alive": "5m", "options": {"num_predict": 1}})
    report["load_s"] = round(time.monotonic() - started, 1)
    report["placement"] = {"where": "unknown"}
    for item in api(endpoint, "/api/ps", 60.0).get("models") or []:
        if isinstance(item, dict) and _canonical_tag(str(item.get("name", ""))) == _canonical_tag(tag):
            placed = placement_of(int(item.get("size") or 0), int(item.get("size_vram") or 0))
            placed["accelerator"] = str(item.get("accelerator") or ("GPU" if item.get("size_vram") else "CPU"))
            report["placement"] = placed
    say("asking two questions")
    messages: list[dict[str, str]] = []
    answers: list[str] = []
    correct = 0
    for question, expected in CHAT:
        messages.append({"role": "user", "content": question})
        reply = api(endpoint, "/api/chat", REQUEST_TIMEOUT, {"model": tag, "messages": messages, "stream": False,
                                                             "options": {"temperature": 0}})
        content = str((reply.get("message") or {}).get("content", ""))
        messages.append({"role": "assistant", "content": content})
        answers.append(content)
        correct += int(expected in content.lower())
    report["chat"] = {"answers": answers, "correct": correct, "of": len(CHAT)}
    say(f"generating {TOKENS} tokens and watching the CPU")
    before = cpu(fork_dir)
    started = time.monotonic()
    reply = api(endpoint, "/api/chat", REQUEST_TIMEOUT, {
        "model": tag, "messages": [{"role": "user", "content": LONG_PROMPT}], "stream": False,
        "options": {"temperature": 0, "num_predict": TOKENS}})
    wall = time.monotonic() - started
    after = cpu(fork_dir)
    eval_count, eval_ns = int(reply.get("eval_count") or 0), int(reply.get("eval_duration") or 0)
    prompt_count, prompt_ns = int(reply.get("prompt_eval_count") or 0), int(reply.get("prompt_eval_duration") or 0)
    words = str((reply.get("message") or {}).get("content", "")).split()
    report["speed"] = {"tokens": eval_count,
                       "decode_tokens_per_s": round(eval_count / (eval_ns / 1e9), 1) if eval_ns else None,
                       "prompt_tokens": prompt_count,
                       "prompt_tokens_per_s": round(prompt_count / (prompt_ns / 1e9), 1) if prompt_ns else None,
                       "words": len(words), "distinct_words": len(set(words))}
    if before is not None and after is not None and after[1] > 0:
        report["cpu"] = {"cores": round(max(0.0, after[0] - before[0]) / max(wall, 1e-3), 2),
                         "processes": after[1], "seconds": round(wall, 1)}
    elif fork_dir is None:
        report["cpu"] = {"skipped": "the fork executable is not known (TEMPERUM_OLLAMA_EXE)"}
    elif sys.platform != "win32":
        report["cpu"] = {"skipped": "measured on Windows only"}
    else:
        report["cpu"] = {"skipped": f"no fork processes found under {fork_dir}"}
    api(endpoint, "/api/generate", 60.0, {"model": tag, "keep_alive": 0})
    return report
