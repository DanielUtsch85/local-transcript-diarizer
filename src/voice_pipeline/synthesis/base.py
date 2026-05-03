from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence


class SynthesisBackend(ABC):
    @abstractmethod
    def synthesize(
        self,
        text: str,
        reference_wavs: Sequence[Path],
        output_wav: Path,
        language: str = "de",
    ) -> Path:
        ...
