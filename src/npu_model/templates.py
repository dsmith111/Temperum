"""Ollama (Go) chat templates for converted models, derived from the model's own Jinja template.

Ollama does not execute Hugging Face Jinja templates (it matches them against a fixed list of
known ones, which modern templates rarely match). The prompt format is recognized from the
Jinja source and a curated Go template reproducing it is emitted, including the model's
default system prompt. Unrecognized formats need an explicit template from the user.

`ChatTemplate.render` mirrors each Go template in Python, so verification can format prompts
exactly as Ollama will and compare them with the model's own template.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from npu_model.core.errors import NpuModelError


@dataclass(frozen=True)
class ChatTemplate:
    family: str
    template: str
    stops: tuple[str, ...]
    default_system: str | None
    verified: bool  # served through the fork and checked end to end
    variant: str = ""  # family detail: chatml/deepseek "think", phi3 "newline"/"compact"

    def modelfile(self) -> str:
        """Modelfile for a bundle folder (FROM is the folder itself)."""
        lines = ["FROM .", f'TEMPLATE """{self.template}"""']
        lines += [f'PARAMETER stop "{stop}"' for stop in self.stops]
        return "\n".join(lines) + "\n"

    def render(self, messages: list[dict[str, str]], think: bool | None = None) -> str | None:
        """The prompt Ollama builds from `messages` with this template (None if unknown).

        `think` mirrors the request's think flag (None: not set).
        """
        system, turns = _collate(messages)
        if self.family == "chatml" and self.variant == "think":
            text = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
            for index, m in enumerate(turns):
                reasoning = m.get("thinking") if index == len(turns) - 1 else None
                block = f"<think>\n{reasoning}\n</think>\n\n" if reasoning else ""
                text += f"<|im_start|>{m['role']}\n{block}{m['content']}<|im_end|>\n"
            return text + "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if think is False else "")
        if self.family in ("chatml", "chatml-imsep"):
            sep, end = ("\n", "\n") if self.family == "chatml" else ("<|im_sep|>", "")
            text = ""
            if system or self.default_system:
                text += f"<|im_start|>system{sep}{system or self.default_system}<|im_end|>{end}"
            text += "".join(f"<|im_start|>{m['role']}{sep}{m['content']}<|im_end|>{end}" for m in turns)
            return text + f"<|im_start|>assistant{sep}"
        if self.family == "llama3":
            text = f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>" if system else ""
            text += "".join(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
                            for m in turns)
            return text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if self.family == "phi3":
            nl = "\n" if self.variant == "newline" else ""
            text = f"<|system|>{nl}{system}<|end|>{nl}" if system else ""
            text += "".join(f"<|{m['role']}|>{nl}{m['content']}<|end|>{nl}" for m in turns)
            return text + f"<|assistant|>{nl}"
        if self.family == "deepseek":
            text = system or ""
            for m in turns:
                if m["role"] == "user":
                    text += f"<｜User｜>{m['content']}"
                elif m["role"] == "assistant":
                    text += f"<｜Assistant｜>{m['content']}<｜end▁of▁sentence｜>"
            return text + "<｜Assistant｜>" + ("<think>\n" if self.variant == "think" else "")
        if self.family == "raw":
            prompt = turns[-1]["content"] if turns else ""
            return (f"{system}\n\n" if system else "") + prompt
        return None


def _collate(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    """Ollama's message collation: system texts joined; consecutive same-role turns merged."""
    systems = [m["content"] for m in messages if m["role"] == "system"]
    merged: list[dict[str, str]] = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1] = {**merged[-1], "content": merged[-1]["content"] + "\n\n" + message["content"]}
        else:
            merged.append(dict(message))
    return ("\n\n".join(systems) or None), [m for m in merged if m["role"] != "system"]


def _chatml(default_system: str | None, separator: str = "\n", after_end: str = "\n") -> str:
    system_default = (f"{{{{ else }}}}<|im_start|>system{separator}{default_system}<|im_end|>{after_end}"
                      if default_system else "")
    return ("{{- if .System }}<|im_start|>system" + separator + "{{ .System }}<|im_end|>" + after_end
            + system_default + "{{ end }}"
            "{{- range .Messages }}{{- if ne .Role \"system\" }}<|im_start|>{{ .Role }}" + separator
            + "{{ .Content }}<|im_end|>" + after_end + "{{ end }}{{- end }}<|im_start|>assistant" + separator)


# Qwen3-style ChatML with reasoning. Rendering `.Thinking` inside the message loop between
# <think> tags is what lets the server infer the thinking capability (thinking.InferTags);
# `think: false` adds the empty think block that `enable_thinking=False` produces.
_CHATML_THINK = (
    "{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}"
    "{{- range $i, $_ := .Messages }}"
    "{{- $last := eq (len (slice $.Messages $i)) 1 -}}"
    "{{- if ne .Role \"system\" }}<|im_start|>{{ .Role }}\n"
    "{{ if and $last .Thinking }}<think>\n{{ .Thinking }}\n</think>\n\n{{ end }}"
    "{{ .Content }}<|im_end|>\n{{ end }}{{- end }}<|im_start|>assistant\n"
    "{{ if and .IsThinkSet (not .Think) }}<think>\n\n</think>\n\n{{ end }}")

_LLAMA3 = ("{{- if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}"
           "{{- range .Messages }}{{- if ne .Role \"system\" }}<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n"
           "{{ .Content }}<|eot_id|>{{ end }}{{- end }}<|start_header_id|>assistant<|end_header_id|>\n\n")


def _phi3(newline: bool) -> str:
    nl = "\n" if newline else ""
    return ("{{- if .System }}<|system|>" + nl + "{{ .System }}<|end|>" + nl + "{{ end }}"
            "{{- range .Messages }}{{- if ne .Role \"system\" }}<|{{ .Role }}|>" + nl + "{{ .Content }}<|end|>"
            + nl + "{{ end }}{{- end }}<|assistant|>" + nl)


_DEEPSEEK = ("{{- if .System }}{{ .System }}{{ end }}"
             "{{- range .Messages }}{{- if eq .Role \"user\" }}<｜User｜>{{ .Content }}"
             "{{- else if eq .Role \"assistant\" }}<｜Assistant｜>{{ .Content }}<｜end▁of▁sentence｜>"
             "{{- end }}{{- end }}<｜Assistant｜>")
_DEEPSEEK_THINK = re.compile(r"<｜Assistant｜><think>(?:\\n|\n)")

_RAW = "{{- if .System }}{{ .System }}\n\n{{ end }}{{ .Prompt }}"

_CHATML_SYSTEM = re.compile(r"<\|im_start\|>system(?:\\n|\n)([^'\"{}<>+\\]+?)<\|im_end\|>")


def load_jinja(model_dir: Path) -> str | None:
    """The model's chat template: chat_template.jinja, else tokenizer_config.json."""
    standalone = model_dir / "chat_template.jinja"
    if standalone.is_file():
        return standalone.read_text(encoding="utf-8")
    try:
        config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    template = config.get("chat_template")
    if isinstance(template, list):  # named templates; use the default one
        named = {item.get("name"): item.get("template") for item in template if isinstance(item, dict)}
        template = named.get("default")
    return template if isinstance(template, str) and template.strip() else None


def from_jinja(jinja: str | None) -> ChatTemplate:
    if jinja is None:
        return ChatTemplate("raw", _RAW, (), None, verified=False)
    if "<｜User｜>" in jinja and "<｜Assistant｜>" in jinja:
        think = bool(_DEEPSEEK_THINK.search(jinja))
        return ChatTemplate("deepseek", _DEEPSEEK + ("<think>\n" if think else ""),
                            ("<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>"), None, verified=False,
                            variant="think" if think else "")
    if "<|start_header_id|>" in jinja and "<|eot_id|>" in jinja:
        return ChatTemplate("llama3", _LLAMA3, ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"),
                            None, verified=False)
    if "<|im_start|>" in jinja and "<|im_end|>" in jinja:
        if "<|im_sep|>" in jinja:  # Phi-4 style: role<|im_sep|>content<|im_end|>, no newlines
            return ChatTemplate("chatml-imsep", _chatml(None, "<|im_sep|>", ""), ("<|im_start|>", "<|im_end|>"),
                                None, verified=False)
        if "enable_thinking" in jinja and "<think>" in jinja:  # Qwen3-style reasoning switch
            return ChatTemplate("chatml", _CHATML_THINK, ("<|im_start|>", "<|im_end|>"), None, verified=False,
                                variant="think")
        match = _CHATML_SYSTEM.search(jinja)
        default = match.group(1).strip() if match else None
        return ChatTemplate("chatml", _chatml(default), ("<|im_start|>", "<|im_end|>"), default, verified=True)
    if "<|user|>" in jinja and "<|end|>" in jinja and "<|assistant|>" in jinja:
        newline = bool(re.search(r"\|>'\s*\+\s*'\\n'|\|>\\n|\|>\n", jinja))
        return ChatTemplate("phi3", _phi3(newline), ("<|end|>", "<|user|>", "<|assistant|>", "<|endoftext|>"),
                            None, verified=False, variant="newline" if newline else "compact")
    raise NpuModelError("template", "TEMPLATE_UNKNOWN",
                        "The model's chat template is not a recognized prompt format.",
                        hint="Pass --template FILE with an Ollama Go template (and --stop for its end markers).")


def custom(template_text: str, stops: list[str]) -> ChatTemplate:
    if "{%" in template_text or '"""' in template_text or not re.search(r"\{\{[^{}]*\.(Prompt|Messages)\b", template_text):
        raise NpuModelError("template", "BAD_TEMPLATE",
                            "--template must be an Ollama Go template using .Messages or .Prompt (not Jinja).")
    return ChatTemplate("custom", template_text, tuple(stops), None, verified=False)
