"""Where the tool keeps toolchains, model snapshots, jobs and bundles.

Roots stay short on purpose: Windows long paths are often disabled, and Olive caches and
Python environments nest deeply. Override with NPU_MODEL_HOME.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def home() -> Path:
    override = os.environ.get("NPU_MODEL_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "npu-model"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "npu-model"


def envs_dir() -> Path:
    return home() / "envs"


def models_dir() -> Path:
    return home() / "models"


def data_dir() -> Path:
    return home() / "data"


def jobs_dir() -> Path:
    return home() / "jobs"


def bundles_dir() -> Path:
    return home() / "bundles"
