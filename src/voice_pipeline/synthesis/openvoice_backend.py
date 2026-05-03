from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .base import SynthesisBackend


class OpenVoiceBackend(SynthesisBackend):
    """OpenVoice V2 adapter placeholder with lazy dependency checks.

    OpenVoice setups vary by checkpoint layout. This class keeps the production
    boundary stable while avoiding heavyweight imports during normal tests.
    """

    def __init__(self, checkpoint_dir: Path | None = None) -> None:
        self.checkpoint_dir = checkpoint_dir

    def synthesize(
        self,
        text: str,
        reference_wavs: Sequence[Path],
        output_wav: Path,
        language: str = "de",
    ) -> Path:
        try:
            import openvoice  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Install OpenVoice V2 and configure checkpoints to use the openvoice backend"
            ) from exc
        raise NotImplementedError(
            "OpenVoice V2 is detected, but checkpoint-specific synthesis wiring must be configured"
        )
