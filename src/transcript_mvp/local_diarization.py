from __future__ import annotations

from pathlib import Path

from .models import SpeakerSegment


def run_local_diarize(
    audio_path: Path,
    min_speakers: int | None,
    max_speakers: int | None,
) -> list[SpeakerSegment]:
    try:
        from diarize import diarize
    except ImportError as exc:
        raise RuntimeError(
            "Das lokale Sprecher-Backend `diarize` ist nicht installiert. "
            "Bitte `python -m pip install diarize` ausfuehren."
        ) from exc

    minimum = min_speakers or 1
    maximum = max_speakers or max(20, minimum)
    exact = minimum if min_speakers and max_speakers and min_speakers == max_speakers else None

    result = diarize(
        audio_path,
        min_speakers=minimum,
        max_speakers=maximum,
        num_speakers=exact,
    )

    return [
        SpeakerSegment(start=float(segment.start), end=float(segment.end), speaker=str(segment.speaker))
        for segment in result.segments
    ]
