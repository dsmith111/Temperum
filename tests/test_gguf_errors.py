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
