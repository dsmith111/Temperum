"""Known quirks in Hugging Face checkpoints that break the pinned toolchains, repaired in a view of the snapshot.

The snapshot stays byte-identical to its pinned commit. When a repair applies, the conversion reads a sibling
folder in the job where every file is a hard link to the snapshot's (a copy where linking is impossible) except
the few that are rewritten. Each repair is small, named, and recorded in the bundle's provenance.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

Repair = Callable[[dict[str, Any]], "dict[str, Any] | None"]


def extra_special_tokens_as_list(config: dict[str, Any]) -> dict[str, Any] | None:
    """transformers 4.57 reads `extra_special_tokens` as a mapping (attribute name to token); checkpoints saved
    by other versions store a plain list of tokens there, and the tokenizer dies with `'list' object has no
    attribute 'keys'`. A list of tokens belongs in `additional_special_tokens`."""
    value = config.get("extra_special_tokens")
    if not isinstance(value, list):
        return None
    fixed = dict(config)
    existing = list(fixed.get("additional_special_tokens") or [])
    fixed["additional_special_tokens"] = existing + [token for token in value if token not in existing]
    del fixed["extra_special_tokens"]
    return fixed


# (file in the snapshot, repair, what it did) in the order they are tried.
REPAIRS: tuple[tuple[str, Repair, str], ...] = (
    ("tokenizer_config.json", extra_special_tokens_as_list,
     "extra_special_tokens was a list of tokens (transformers 4.57 expects a mapping); moved to "
     "additional_special_tokens"),
)


def planned(snapshot: Path) -> dict[str, tuple[dict[str, Any], str]]:
    """The files a snapshot needs rewritten: name -> (new content, what changed)."""
    patched: dict[str, tuple[dict[str, Any], str]] = {}
    for name, repair, description in REPAIRS:
        path = snapshot / name
        if name in patched or not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        fixed = repair(data)
        if fixed is not None:
            patched[name] = (fixed, description)
    return patched


def prepare_source(snapshot: Path, view: Path, say: Callable[[str], None]) -> tuple[Path, list[str]]:
    """The folder a conversion should read: the snapshot itself, or a hard-linked view with repairs applied.
    Returns it and the list of repairs made (for the provenance)."""
    patched = planned(snapshot)
    if not patched:
        return snapshot, []
    if view.exists():
        shutil.rmtree(view)
    view.mkdir(parents=True)
    for item in sorted(snapshot.rglob("*")):
        relative = item.relative_to(snapshot)
        if any(part.startswith(".") for part in relative.parts):  # the hub's own .cache metadata
            continue
        target = view / relative
        if item.is_dir():
            target.mkdir(exist_ok=True)
        elif len(relative.parts) == 1 and relative.name in patched:
            target.write_text(json.dumps(patched[relative.name][0], indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
        else:
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)
    applied = [f"{name}: {description}" for name, (_content, description) in patched.items()]
    say("repaired the source for the toolchain (the download itself is untouched): " + "; ".join(applied))
    return view, applied
