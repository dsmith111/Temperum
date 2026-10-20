"""Where the tool keeps toolchains, model snapshots, jobs and bundles.

Roots stay short on purpose: Windows long paths are often disabled, and Olive caches and
Python environments nest deeply. Override with TEMPERUM_HOME.

  <home>/envs/<runtime>/<role>   toolchain environments
  <home>/python                  uv-managed interpreters
  <home>/sources/<org>--<name>/<commit>   pinned source snapshots
  <home>/data                    calibration and held-out text
  <home>/jobs/<id>               conversion jobs (config, logs, Olive cache, receipt)
  <home>/models/<name>-<target>  converted bundles
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def home() -> Path:
    override = os.environ.get("TEMPERUM_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "temperum"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "temperum"


def models_dir() -> Path:
    """Converted bundles."""
    return home() / "models"


def sources_dir() -> Path:
    """Pinned source-model snapshots."""
    return home() / "sources"


def data_dir() -> Path:
    return home() / "data"


def jobs_dir() -> Path:
    return home() / "jobs"
