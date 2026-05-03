from __future__ import annotations

from pathlib import Path
from typing import Sequence

from voice_pipeline.audio_io import create_dummy_wav

from .base import SynthesisBackend


class MockBackend(SynthesisBackend):
    def synthesize(
        self,
        text: str,
        reference_wavs: Sequence[Path],
        output_wav: Path,
        language: str = "de",
    ) -> Path:
        return create_dummy_wav(output_wav, duration_sec=1.0, sample_rate=24000, tone_hz=440.0)
