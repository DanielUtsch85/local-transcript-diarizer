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

    try:
        result = diarize(
            audio_path,
            min_speakers=minimum,
            max_speakers=maximum,
            num_speakers=exact,
        )
    except RuntimeError as exc:
        raise RuntimeError(format_local_diarize_error(str(exc))) from exc

    return [
        SpeakerSegment(start=float(segment.start), end=float(segment.end), speaker=str(segment.speaker))
        for segment in result.segments
    ]


def format_local_diarize_error(details: str) -> str:
    if is_corrupt_silero_vad_error(details):
        return (
            "Die Transkription wurde erstellt, aber die lokale Sprechererkennung ist beim Laden von Silero VAD "
            "gescheitert.\n\n"
            "Die installierte Silero-Modelldatei scheint beschaedigt oder unlesbar zu sein. Bitte `silero-vad` "
            "in der virtuellen Umgebung neu installieren, oder den Lauf vorerst mit `Nur transkribieren` starten.\n\n"
            f"{details}"
        )
    return f"Lokale Sprechererkennung fehlgeschlagen.\n\n{details}"


def is_corrupt_silero_vad_error(details: str) -> bool:
    markers = [
        "PytorchStreamReader failed reading zip archive",
        "failed finding central directory",
    ]
    return all(marker in details for marker in markers)
