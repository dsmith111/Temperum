"""Temperum's patches to the pinned llama.cpp converter: applied after the archive is verified, checked after."""
from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from typing import Any

import pytest

from temperum.core.errors import TemperumError
from temperum.toolchains import llama_cpp
from temperum.toolchains.llama_cpp import ConverterSource


def _archive(files: dict[str, bytes]) -> Any:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo("llama.cpp-abc/" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    return lambda _url: buffer


def test_patches_replace_pinned_files_after_the_archive_is_verified(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEMPERUM_HOME", str(tmp_path / "home"))
    patch = tmp_path / "gpt2.py"
    patch.write_text("print('fixed')\n", encoding="utf-8")
    source = ConverterSource("t", "abc", "https://example.invalid/a.tar.gz",
                             {"convert_hf_to_gguf.py": hashlib.sha256(b"print(1)\n").hexdigest(),
                              "conversion/gpt2.py": hashlib.sha256(b"print('broken')\n").hexdigest()},
                             {"conversion/gpt2.py": patch})
    said: list[str] = []
    source.ensure(opener=_archive({"convert_hf_to_gguf.py": b"print(1)\n", "conversion/gpt2.py": b"print('broken')\n"}),
                  progress=said.append)
    assert (source.directory / "conversion" / "gpt2.py").read_text(encoding="utf-8") == "print('fixed')\n"
    assert source.problems() == [] and source.problems(upstream=True) == ["conversion/gpt2.py"]
    assert any("converter patches: conversion/gpt2.py" in line for line in said)
    # the archive itself must still match upstream: a patched file arriving in the archive is a mismatch
    with pytest.raises(Exception, match="LLAMA_CPP_MISMATCH|does not match"):
        ConverterSource("u", "abc", "https://example.invalid/a.tar.gz",
                        {"conversion/gpt2.py": hashlib.sha256(b"print('broken')\n").hexdigest()},
                        {"conversion/gpt2.py": patch}).ensure(opener=_archive({"conversion/gpt2.py": b"print('fixed')\n"}))


def test_bytecode_pip_compiles_next_to_a_patch_is_not_a_patch(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    # `pip install .` byte-compiles the package, so the installed patch folder gains a __pycache__ beside
    # gpt2.py; copying that into the fetched converter failed (no such directory) on the owner's install
    root = tmp_path / "patches" / "b11644" / "conversion"
    (root / "__pycache__").mkdir(parents=True)
    (root / "gpt2.py").write_text("print('fixed')\n", encoding="utf-8")
    (root / "__pycache__" / "gpt2.cpython-314.pyc").write_bytes(b"\x00")
    monkeypatch.setattr(llama_cpp, "PATCHES", tmp_path / "patches")
    assert list(ConverterSource.load("b11644").patches) == ["conversion/gpt2.py"]


def test_a_patch_must_name_a_pinned_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "patches" / "b11644"
    root.mkdir(parents=True)
    (root / "convert_hf_to_gguf_typo.py").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(llama_cpp, "PATCHES", tmp_path / "patches")
    with pytest.raises(TemperumError, match="names no file pinned"):
        ConverterSource.load("b11644")


def test_the_pinned_converter_carries_the_gpt2_patch() -> None:
    source = ConverterSource.load("b11644")
    assert "conversion/gpt2.py" in source.patches
    text = source.patches["conversion/gpt2.py"].read_text(encoding="utf-8")
    assert "Temperum patch" in text and 'name.endswith((".attn.bias", ".attn.masked_bias")):\n            return' in text
    assert source.expected("conversion/gpt2.py") != source.files["conversion/gpt2.py"]
    assert source.expected("conversion/gpt2.py", upstream=True) == source.files["conversion/gpt2.py"]
