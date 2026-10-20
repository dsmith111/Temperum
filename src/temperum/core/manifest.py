from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_files(root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            items.append(
                {
                    "path": str(p.relative_to(root)).replace("\\", "/"),
                    "size": p.stat().st_size,
                    "sha256": sha256_file(p),
                }
            )
    return items


class _ManifestEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, cls=_ManifestEncoder), encoding="utf-8")
