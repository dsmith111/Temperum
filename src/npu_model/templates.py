"""Ollama (Go) chat templates for converted models, derived from the model's own Jinja template.

Ollama does not execute Hugging Face Jinja templates (it matches them against a fixed list of
known ones, which modern templates rarely match). The prompt format is recognized from the
Jinja source and a curated Go template reproducing it is emitted, including the model's
default system prompt. Unrecognized formats need an explicit template from the user.
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

    def modelfile(self) -> str:
        """Modelfile for a bundle folder (FROM is the folder itself)."""
        lines = ["FROM .", f'TEMPLATE """{self.template}"""']
        lines += [f'PARAMETER stop "{stop}"' for stop in self.stops]
        return "\n".join(lines) + "\n"


def _chatml(default_system: str | None, separator: str = "\n", after_end: str = "\n") -> str:
    system_default = (f"{{{{ else }}}}<|im_start|>system{separator}{default_system}<|im_end|>{after_end}"
                      if default_system else "")
    return ("{{- if .System }}<|im_start|>system" + separator + "{{ .System }}<|im_end|>" + after_end
            + system_default + "{{ end }}"
            "{{- range .Messages }}{{- if ne .Role \"system\" }}<|im_start|>{{ .Role }}" + separator
            + "{{ .Content }}<|im_end|>" + after_end + "{{ end }}{{- end }}<|im_start|>assistant" + separator)


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
        template = _DEEPSEEK + ("<think>\n" if _DEEPSEEK_THINK.search(jinja) else "")
        return ChatTemplate("deepseek", template, ("<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>"),
                            None, verified=False)
    if "<|start_header_id|>" in jinja and "<|eot_id|>" in jinja:
        return ChatTemplate("llama3", _LLAMA3, ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"),
                            None, verified=False)
    if "<|im_start|>" in jinja and "<|im_end|>" in jinja:
        if "<|im_sep|>" in jinja:  # Phi-4 style: role<|im_sep|>content<|im_end|>, no newlines
            return ChatTemplate("chatml-imsep", _chatml(None, "<|im_sep|>", ""), ("<|im_start|>", "<|im_end|>"),
                                None, verified=False)
        match = _CHATML_SYSTEM.search(jinja)
        default = match.group(1).strip() if match else None
        return ChatTemplate("chatml", _chatml(default), ("<|im_start|>", "<|im_end|>"), default, verified=True)
    if "<|user|>" in jinja and "<|end|>" in jinja and "<|assistant|>" in jinja:
        newline = bool(re.search(r"\|>'\s*\+\s*'\\n'|\|>\\n|\|>\n", jinja))
        return ChatTemplate("phi3", _phi3(newline), ("<|end|>", "<|user|>", "<|assistant|>", "<|endoftext|>"),
                            None, verified=False)
    raise NpuModelError("template", "TEMPLATE_UNKNOWN",
                        "The model's chat template is not a recognized prompt format.",
                        hint="Pass --template FILE with an Ollama Go template (and --stop for its end markers).")


def custom(template_text: str, stops: list[str]) -> ChatTemplate:
    if "{%" in template_text or '"""' in template_text or not re.search(r"\{\{[^{}]*\.(Prompt|Messages)\b", template_text):
        raise NpuModelError("template", "BAD_TEMPLATE",
                            "--template must be an Ollama Go template using .Messages or .Prompt (not Jinja).")
    return ChatTemplate("custom", template_text, tuple(stops), None, verified=False)
