"""Settings that outlive a terminal: the fork executable, the default target and the server endpoint.

They live in `config.json` in the temperum home. An option on the command line wins, then the
environment variable (`TEMPERUM_OLLAMA_EXE`, `TEMPERUM_TARGET`, `OLLAMA_HOST`), then the file.
`temperum setup --ollama-exe` and `temperum config` write the file; nothing else does. A value set with
`setx` only reaches terminals started afterwards (and none inside an editor started before), which is
why the file exists.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from temperum import paths
from temperum.core.errors import TemperumError

FILE = "config.json"
KEYS: dict[str, str] = {"ollama-exe": "TEMPERUM_OLLAMA_EXE", "target": "TEMPERUM_TARGET", "host": "OLLAMA_HOST"}
DESCRIPTIONS: dict[str, str] = {"ollama-exe": "the fork's ollama.exe",
                                "target": "default --target (auto, gguf-hexagon, qnn-x-elite)",
                                "host": "the fork's endpoint, http://127.0.0.1:PORT"}


@dataclass(frozen=True)
class Setting:
    value: str | None
    source: str  # "option", the environment variable's name, "config" or "default"


def path() -> Path:
    return paths.home() / FILE


def load() -> dict[str, str]:
    try:
        data = json.loads(path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise TemperumError("config", "CONFIG_UNREADABLE", f"{path()} cannot be read as JSON: {exc}",
                            hint="Fix or delete the file; `temperum config` rewrites it.") from exc
    if not isinstance(data, dict):
        raise TemperumError("config", "CONFIG_UNREADABLE", f"{path()} must hold a JSON object.",
                            hint="Delete the file; `temperum config` rewrites it.")
    return {key: str(value) for key, value in data.items() if key in KEYS and isinstance(value, str)}


def save(values: dict[str, str]) -> None:
    path().parent.mkdir(parents=True, exist_ok=True)
    path().write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def check_key(key: str) -> str:
    if key not in KEYS:
        raise TemperumError("config", "UNKNOWN_SETTING", f"{key!r} is not a setting.",
                            hint="Settings: " + ", ".join(KEYS))
    return key


def set_value(key: str, value: str) -> None:
    values = load()
    values[check_key(key)] = value
    save(values)


def unset(key: str) -> None:
    values = load()
    values.pop(check_key(key), None)
    save(values)


def resolve(key: str, option: str | None = None, default: str | None = None) -> Setting:
    """The value in effect for a setting and where it came from."""
    check_key(key)
    if option:
        return Setting(option, "option")
    env = os.environ.get(KEYS[key])
    if env:
        return Setting(env, KEYS[key])
    stored = load().get(key)
    if stored:
        return Setting(stored, "config")
    return Setting(default, "default")


def ollama_exe(option: Path | None = None) -> tuple[Path | None, str]:
    """The fork executable in effect (any source) and that source."""
    setting = resolve("ollama-exe", str(option) if option else None)
    return (Path(setting.value) if setting.value else None), setting.source
