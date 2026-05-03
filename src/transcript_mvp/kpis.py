from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .models import TranscriptSegment


@dataclass(frozen=True)
class TranscriptKpis:
    segment_count: int
    speaker_count: int
    word_count: int
    avg_segment_duration: float | None
    max_segment_duration: float | None
    segments_per_hour: float | None
    dominant_speaker_share: float | None
    unknown_speaker_share: float | None


@dataclass(frozen=True)
class ResourceKpis:
    samples: int
    peak_cpu_percent: float | None
    avg_cpu_percent: float | None
    peak_memory_pressure_percent: float | None
    avg_memory_pressure_percent: float | None
    peak_process_ram_gb: float | None
    avg_process_ram_gb: float | None


def transcript_kpis(segments: list[TranscriptSegment], audio_seconds: float | None) -> TranscriptKpis:
    speakers = {segment.speaker for segment in segments}
    durations = [
        max(0.0, segment.end - segment.start)
        for segment in segments
        if segment.start is not None and segment.end is not None
    ]
    speaker_durations: dict[str, float] = defaultdict(float)
    for segment in segments:
        if segment.start is not None and segment.end is not None:
            speaker_durations[segment.speaker] += max(0.0, segment.end - segment.start)

    total_speaker_duration = sum(speaker_durations.values())
    dominant_share = None
    unknown_share = None
    if total_speaker_duration > 0:
        dominant_share = max(speaker_durations.values()) / total_speaker_duration
        unknown_share = speaker_durations.get("SPEAKER_UNKNOWN", 0.0) / total_speaker_duration

    return TranscriptKpis(
        segment_count=len(segments),
        speaker_count=len(speakers),
        word_count=sum(len(segment.text.split()) for segment in segments),
        avg_segment_duration=_avg(durations),
        max_segment_duration=max(durations) if durations else None,
        segments_per_hour=_per_hour(len(segments), audio_seconds),
        dominant_speaker_share=dominant_share,
        unknown_speaker_share=unknown_share,
    )


def resource_kpis(resource_history: list[dict]) -> ResourceKpis:
    return ResourceKpis(
        samples=len(resource_history),
        peak_cpu_percent=_max_key(resource_history, "CPU gesamt %"),
        avg_cpu_percent=_avg_key(resource_history, "CPU gesamt %"),
        peak_memory_pressure_percent=_max_key(resource_history, "Speicherdruck %"),
        avg_memory_pressure_percent=_avg_key(resource_history, "Speicherdruck %"),
        peak_process_ram_gb=_max_key(resource_history, "WhisperX RAM GB"),
        avg_process_ram_gb=_avg_key(resource_history, "WhisperX RAM GB"),
    )


def quality_notes(kpis: TranscriptKpis) -> list[str]:
    notes = []
    if kpis.segment_count == 0:
        notes.append("Kein Transkript erzeugt.")
    if kpis.speaker_count <= 1:
        notes.append("Nur ein Sprecher erkannt. Bei Interviews bitte Ergebnis pruefen.")
    if kpis.max_segment_duration and kpis.max_segment_duration > 180:
        notes.append("Mindestens ein Abschnitt ist sehr lang. Das Word-Dokument kann schwer lesbar sein.")
    if kpis.dominant_speaker_share and kpis.dominant_speaker_share > 0.9:
        notes.append("Ein Sprecher dominiert stark. Das kann bei Interviews normal sein, sollte aber geprueft werden.")
    if kpis.unknown_speaker_share and kpis.unknown_speaker_share > 0.05:
        notes.append("Mehr als 5% der Sprecherzeit ist unbekannt.")
    if not notes:
        notes.append("Keine offensichtlichen Strukturprobleme erkannt.")
    return notes


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _per_hour(count: int, seconds: float | None) -> float | None:
    if seconds is None or seconds <= 0:
        return None
    return count / (seconds / 3600)


def _max_key(rows: list[dict], key: str) -> float | None:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    return max(values) if values else None


def _avg_key(rows: list[dict], key: str) -> float | None:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    return _avg(values)
