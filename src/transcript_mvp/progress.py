from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ProgressReporter:
    on_overall: Callable[[float, str], None]
    on_transcription: Callable[[float], None]
    on_tick: Callable[[float, int], None]
    on_log: Callable[[str], None]
    _transcription_value: float = field(default=0.0, init=False)

    def overall(self, value: float, text: str) -> None:
        self.on_overall(min(1.0, max(0.0, value)), text)

    def transcription(self, value: float) -> None:
        self._transcription_value = value
        self.on_transcription(value)

    def tick(self, elapsed: float, pid: int) -> None:
        self.on_tick(elapsed, pid)

    def log(self, line: str) -> None:
        self.on_log(line)
