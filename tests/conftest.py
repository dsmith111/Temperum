from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _hexagon_switch_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--target auto` follows the fork's OLLAMA_HEXAGON switch; tests turn it on explicitly."""
    monkeypatch.delenv("OLLAMA_HEXAGON", raising=False)
