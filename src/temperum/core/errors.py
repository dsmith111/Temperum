from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TemperumError(Exception):
    stage: str
    reason_code: str
    message: str
    hint: Optional[str] = None
    cause: Optional[BaseException] = None

    def __str__(self) -> str:
        return self.message
