"""The converter's failure names the error a person needs first."""
from __future__ import annotations

from temperum import gguf

TAIL = """  File "conversion/base.py", line 1603, in prepare_metadata
    self.set_vocab()
  File "conversion/qwen.py", line 753, in set_vocab
    raise ValueError(
ValueError: DFlash draft model requires --target-model-dir to be specified. Please provide the path to the target model
directory containing the tokenizer.
"""


def test_last_error_line_finds_the_exception_line() -> None:
    assert gguf.last_error_line(TAIL) == ("ValueError: DFlash draft model requires --target-model-dir to be specified. "
                                          "Please provide the path to the target model")
    assert gguf.last_error_line("INFO: all good\nModel successfully exported") is None
    assert gguf.last_error_line("torch.OutOfMemoryError: CUDA out of memory") == \
        "torch.OutOfMemoryError: CUDA out of memory"
    assert gguf.last_error_line("  Error loading x\nwarning: Error-prone") is None  # not exception lines


def test_last_error_line_accepts_module_paths_with_digits() -> None:
    # jinja2.exceptions.TemplateError (K2-Horizon's template refusing an assistant turn without a thinking
    # field) was missed, so the panel showed 400 characters of frames instead of the one line that matters
    tail = ("  File \"utils.py\", line 447, in raise_exception\n    raise jinja2.exceptions.TemplateError(message)\n"
            "jinja2.exceptions.TemplateError: Assistant message is missing a thinking field.")
    assert gguf.last_error_line(tail) == "jinja2.exceptions.TemplateError: Assistant message is missing a thinking field."
