"""Generated templates executed by the Ollama fork's own template engine (opt-in integration test).

Set TEMPERUM_FORK_DIR to a checkout of the fork; needs Go. The checker program is mapped into
the fork module with `go run -overlay`, so the checkout is never modified.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from temperum.templates import from_jinja

FORK = os.environ.get("TEMPERUM_FORK_DIR")
FIXTURES = Path(__file__).parent / "fixtures"
pytestmark = pytest.mark.skipif(not FORK or not shutil.which("go"), reason="set TEMPERUM_FORK_DIR; needs Go")

JINJAS = {
    "qwen2.5": (FIXTURES / "templates" / "qwen2.5-instruct.jinja").read_text(encoding="utf-8"),
    "smollm2": (FIXTURES / "templates" / "smollm2-instruct.jinja").read_text(encoding="utf-8"),
    "qwen3": (FIXTURES / "templates" / "qwen3.jinja").read_text(encoding="utf-8"),
    "llama3": "{{ bos_token }}<|start_header_id|>{{ role }}<|end_header_id|>\n\n{{ content }}<|eot_id|>",
    "phi3-newline": "{{'<|' + message['role'] + '|>' + '\\n' + message['content'] + '<|end|>' + '\\n'}}<|user|><|assistant|>",
    "phi3-compact": "{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}<|user|>{{ '<|assistant|>' }}",
    "deepseek-think": "{{'<｜User｜>' + content}}{{'<｜Assistant｜><think>\\n'}}",
    "phi4": "{{'<|im_start|>user<|im_sep|>' + content + '<|im_end|>'}}",
}
CONVERSATIONS = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"},
     {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "What is 2+2?"}],
    [{"role": "system", "content": "A"}, {"role": "system", "content": "B"}, {"role": "user", "content": "x"},
     {"role": "user", "content": "y"}],
    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello", "thinking": "greet"}],
]


def test_fork_renders_every_generated_template_like_the_python_mirror(tmp_path: Path) -> None:
    assert FORK
    cases, expected = [], []
    for name, jinja in JINJAS.items():
        chat = from_jinja(jinja)
        for conversation in CONVERSATIONS:
            for think in (None, True, False):
                cases.append({"template": chat.template, "messages": conversation, "think": think})
                expected.append((name, chat.render(conversation, think=think)))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    overlay = tmp_path / "overlay.json"
    target = Path(FORK) / "x" / "npucheck" / "main.go"
    overlay.write_text(json.dumps({"Replace": {str(target): str(FIXTURES / "gocheck" / "main.go.txt")}}),
                       encoding="utf-8")
    proc = subprocess.run(["go", "run", "-overlay", str(overlay), "./x/npucheck", str(spec)], cwd=FORK,
                          env={**os.environ, "CGO_ENABLED": "0"}, capture_output=True, text=True,
                          encoding="utf-8", timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    results = json.loads(proc.stdout)
    for (name, ours), got in zip(expected, results):
        assert not got.get("error"), (name, got["error"])
        assert got["rendered"] == ours, name
    tags = {name: (r["open"], r["close"]) for (name, _), r in zip(expected, results)}
    assert tags["qwen3"] == ("<think>", "</think>")  # the server grants the thinking capability
    assert all(value == ("", "") for name, value in tags.items() if name != "qwen3")
