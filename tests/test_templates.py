from __future__ import annotations

import json
from pathlib import Path

import pytest

from npu_model.core.errors import NpuModelError
from npu_model.install.ollama_import import read_modelfile
from npu_model.templates import custom, from_jinja, load_jinja

FIXTURES = Path(__file__).parent / "fixtures" / "templates"


def _jinja(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_qwen25_matches_the_template_served_in_m0() -> None:
    chat = from_jinja(_jinja("qwen2.5-instruct.jinja"))
    assert chat.family == "chatml" and chat.verified
    assert chat.default_system == "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    golden = (FIXTURES / "qwen2.5-m0.gotmpl").read_text(encoding="utf-8")
    assert chat.template == golden if golden.endswith("\n") else chat.template == golden + "\n"
    assert chat.stops == ("<|im_start|>", "<|im_end|>")


def test_smollm2_keeps_its_default_system_prompt() -> None:
    chat = from_jinja(_jinja("smollm2-instruct.jinja"))
    assert chat.family == "chatml"
    assert chat.default_system == "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
    assert "{{ else }}<|im_start|>system\nYou are a helpful AI assistant named SmolLM" in chat.template


def test_qwen3_has_no_default_system_prompt() -> None:
    chat = from_jinja(_jinja("qwen3.jinja"))
    assert chat.family == "chatml" and chat.default_system is None and "{{ else }}" not in chat.template


def test_qwen3_gets_a_thinking_aware_template() -> None:
    chat = from_jinja(_jinja("qwen3.jinja"))
    assert chat.variant == "think" and chat.verified  # served through the fork: thinking split, think=false
    # The server infers the thinking capability from .Thinking between tags inside the message loop.
    assert "{{ if and $last .Thinking }}<think>\n{{ .Thinking }}\n</think>" in chat.template
    hello = [{"role": "user", "content": "Hello!"}]
    # enable_thinking=False in the HF template appends an empty think block after the prompt.
    assert chat.render(hello, think=False) == (
        "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert chat.render(hello, think=True) == chat.render(hello) == (
        "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n")
    prefill = hello + [{"role": "assistant", "content": "Hi", "thinking": "greet back"}]
    assert chat.render(prefill).endswith("<|im_start|>assistant\n<think>\ngreet back\n</think>\n\nHi<|im_end|>\n"
                                         "<|im_start|>assistant\n")


@pytest.mark.parametrize("jinja,family,needle", [
    ("{{ bos_token }}<|start_header_id|>{{ role }}<|end_header_id|>\n\n{{ content }}<|eot_id|>", "llama3",
     "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    ("{{'<|' + message['role'] + '|>' + '\\n' + message['content'] + '<|end|>' + '\\n'}}<|user|><|assistant|>",
     "phi3", "<|assistant|>\n"),
    ("{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}<|user|>{{ '<|assistant|>' }}",
     "phi3", "<|end|>{{ end }}{{- end }}<|assistant|>"),
    # Phi-4-mini-instruct's own template: role tags built from message['role'], no literal <|user|>
    ("{% for message in messages %}{% if message['role'] == 'system' and 'tools' in message and message['tools'] "
     "is not none %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|tool|>' + message['tools'] + "
     "'<|/tool|>' + '<|end|>' }}{% else %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}"
     "{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %}",
     "phi3", "<|end|>{{ end }}{{- end }}<|assistant|>"),
    ("{{'<｜User｜>' + content}}{{'<｜Assistant｜><think>\\n'}}", "deepseek", "<｜Assistant｜><think>\n"),
    ("{{'<|im_start|>user<|im_sep|>' + content + '<|im_end|>'}}", "chatml-imsep", "<|im_start|>assistant<|im_sep|>"),
])
def test_other_families_are_recognized(jinja: str, family: str, needle: str) -> None:
    chat = from_jinja(jinja)
    assert chat.family == family and not chat.verified
    assert chat.template.endswith(needle)


def test_base_models_without_a_chat_template_use_the_raw_prompt() -> None:
    chat = from_jinja(None)
    assert chat.family == "raw" and ".Prompt" in chat.template and chat.stops == ()


def test_unknown_formats_need_an_explicit_template() -> None:
    with pytest.raises(NpuModelError) as exc:
        from_jinja("{% for m in messages %}### {{ m.role }}: {{ m.content }}{% endfor %}")
    assert exc.value.reason_code == "TEMPLATE_UNKNOWN"


def test_custom_templates_must_be_go_templates() -> None:
    assert custom("{{ .Prompt }}", ["</s>"]).stops == ("</s>",)
    with pytest.raises(NpuModelError):
        custom("{% for m in messages %}{{ m }}{% endfor %}", [])


def test_load_jinja_prefers_the_standalone_file(tmp_path: Path) -> None:
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": [
        {"name": "tool_use", "template": "x"}, {"name": "default", "template": "<|im_start|>"}]}), encoding="utf-8")
    assert load_jinja(tmp_path) == "<|im_start|>"
    (tmp_path / "chat_template.jinja").write_text("standalone", encoding="utf-8")
    assert load_jinja(tmp_path) == "standalone"
    assert load_jinja(tmp_path / "missing") is None


CONVERSATION = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "What is 2+2?"}]


def test_render_matches_what_the_hf_templates_produce() -> None:
    qwen = from_jinja(_jinja("qwen2.5-instruct.jinja"))
    assert qwen.render(CONVERSATION) == (
        "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\nHello.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n")
    smol = from_jinja(_jinja("smollm2-instruct.jinja"))
    assert smol.render([{"role": "user", "content": "Hello!"}]) == (
        "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
        "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n")
    qwen3 = from_jinja(_jinja("qwen3.jinja"))
    assert qwen3.render([{"role": "user", "content": "Hello!"}]) == (
        "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n")


def test_render_follows_ollama_collation() -> None:
    chat = from_jinja(_jinja("qwen3.jinja"))
    merged = chat.render([{"role": "system", "content": "A"}, {"role": "system", "content": "B"},
                          {"role": "user", "content": "x"}, {"role": "user", "content": "y"}])
    assert merged == "<|im_start|>system\nA\n\nB<|im_end|>\n<|im_start|>user\nx\n\ny<|im_end|>\n<|im_start|>assistant\n"
    assert custom("{{ .Prompt }}", []).render([{"role": "user", "content": "x"}]) is None


def test_generated_modelfile_passes_the_install_contract(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("qwen2.5-instruct.jinja", "smollm2-instruct.jinja", "qwen3.jinja"):
        (bundle / "Modelfile").write_text(from_jinja(_jinja(name)).modelfile(), encoding="utf-8", newline="\n")
        parsed = read_modelfile(bundle / "Modelfile", bundle.resolve(), 4096)
        assert parsed.text.startswith("FROM .\nTEMPLATE \"\"\"")
