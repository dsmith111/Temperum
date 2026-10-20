"""Olive user_script: calibration data without the `datasets` package.

Registers a JSONL text loader and a text-generation pre-processor mirroring the "join" and
"line-by-line" strategies of Olive's HuggingfaceContainer, returning Olive's
ClassificationDataset over plain dicts of torch tensors instead of a `datasets.Dataset`
(pyarrow, which `datasets` needs, has no Windows ARM64 build).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from olive.data.component.dataset import ClassificationDataset
from olive.data.registry import Registry


@Registry.register_dataset()
def npu_jsonl_text(data_files: str | list[str], text_field: str = "text", **kwargs: Any) -> list[str]:
    files = data_files if isinstance(data_files, (list, tuple)) else [data_files]
    rows: list[str] = []
    for name in files:
        with Path(name).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line).get(text_field) or "")
    return rows


def _sample(input_ids: list[int], pad_id: int, max_seq_len: int, pad: bool) -> dict[str, Any]:
    ids = input_ids[:max_seq_len]
    mask = [1] * len(ids)
    if pad and len(ids) < max_seq_len:
        missing = max_seq_len - len(ids)
        ids = ids + [pad_id] * missing
        mask = mask + [0] * missing
    ids_t = torch.tensor(ids, dtype=torch.long)
    mask_t = torch.tensor(mask, dtype=torch.long)
    labels = ids_t.clone()
    labels[mask_t == 0] = -100
    return {"input_ids": ids_t, "attention_mask": mask_t, "labels": labels}


@Registry.register_pre_process()
def npu_text_gen(
    dataset: list[str],
    model_name: str,
    strategy: str = "join",
    max_seq_len: int = 1024,
    max_samples: int = 32,
    add_special_tokens: bool = False,
    joiner: str = "\n\n",
    pad_to_max_len: bool = True,
    min_tokens: int = 8,
    random_seed: int = 0,
    **kwargs: Any,
) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    samples: list[dict[str, Any]] = []
    texts = [t for t in dataset if t and t.strip()]

    if strategy == "join":
        joiner_ids = tok.encode(joiner, add_special_tokens=False) if joiner else []
        buf: list[int] = []
        for text in texts:
            buf += tok.encode(text, add_special_tokens=add_special_tokens) + joiner_ids
            while len(buf) >= max_seq_len and len(samples) < max_samples:
                samples.append(_sample(buf[:max_seq_len], pad_id, max_seq_len, False))
                buf = buf[max_seq_len:]
            if len(samples) >= max_samples:
                break
    elif strategy in ("line-by-line", "line-by-line-random"):
        order = list(range(len(texts)))
        if strategy.endswith("random"):
            random.Random(random_seed).shuffle(order)
        for i in order:
            ids = tok.encode(texts[i], add_special_tokens=add_special_tokens)
            if len(ids) < min_tokens:
                continue
            samples.append(_sample(ids, pad_id, max_seq_len, pad_to_max_len))
            if len(samples) >= max_samples:
                break
    else:
        raise ValueError(f"unsupported strategy: {strategy}")

    if not samples:
        raise ValueError("calibration corpus produced no samples")
    return ClassificationDataset(samples, "labels", max_samples=max_samples)
