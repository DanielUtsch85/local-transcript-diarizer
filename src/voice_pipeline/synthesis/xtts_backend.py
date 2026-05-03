from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .base import SynthesisBackend


class XTTSBackend(SynthesisBackend):
    """Adapter for Coqui XTTS-v2.

    The dependency is imported lazily so tests and curation workflows do not
    require model downloads, CUDA, or large checkpoints.
    """

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2") -> None:
        self.model_name = model_name
        self._tts = None

    def _load(self):
        if self._tts is None:
            try:
                from TTS.api import TTS  # type: ignore
            except ImportError as exc:
                raise RuntimeError("Install coqui-tts/TTS to use the XTTS backend") from exc
            self._tts = TTS(self.model_name)
        return self._tts

    def synthesize(
        self,
        text: str,
        reference_wavs: Sequence[Path],
        output_wav: Path,
        language: str = "de",
    ) -> Path:
        if not reference_wavs:
            raise ValueError("XTTS synthesis requires at least one reference wav")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        tts = self._load()
        speaker_wav = [str(path) for path in reference_wavs]
        tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=str(output_wav))
        return output_wav
