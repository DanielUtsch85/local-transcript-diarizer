from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .local_diarization import run_local_diarize
from .models import SpeakerSegment, TranscriptSegment
from .pipeline import assign_speakers_by_overlap


DiarizeFn = Callable[[Path, int | None, int | None], list[SpeakerSegment]]


@dataclass(frozen=True)
class LocalDiarizationOutcome:
    segments: list[TranscriptSegment]
    speaker_segments: list[SpeakerSegment]
    speaker_segments_count: int | None
    error: str | None
    unexpected_error: bool = False


def apply_optional_local_diarization(
    transcript_segments: list[TranscriptSegment],
    audio_path: Path,
    *,
    min_speakers: int | None,
    max_speakers: int | None,
    diarize_fn: DiarizeFn = run_local_diarize,
) -> LocalDiarizationOutcome:
    try:
        speaker_segments = diarize_fn(audio_path, min_speakers, max_speakers)
    except RuntimeError as exc:
        return LocalDiarizationOutcome(
            segments=transcript_segments,
            speaker_segments=[],
            speaker_segments_count=None,
            error=str(exc),
        )
    except Exception as exc:
        return LocalDiarizationOutcome(
            segments=transcript_segments,
            speaker_segments=[],
            speaker_segments_count=None,
            error=format_unexpected_diarization_error(exc),
            unexpected_error=True,
        )

    assigned_segments = assign_speakers_by_overlap(transcript_segments, speaker_segments)
    return LocalDiarizationOutcome(
        segments=assigned_segments,
        speaker_segments=speaker_segments,
        speaker_segments_count=len(speaker_segments),
        error=None,
    )


def format_unexpected_diarization_error(exc: Exception) -> str:
    return (
        "Unerwarteter Fehler in der lokalen Sprechererkennung: "
        f"{type(exc).__name__}: {exc}"
    )
