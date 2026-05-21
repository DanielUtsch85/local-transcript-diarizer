from __future__ import annotations

import csv
from collections import Counter, defaultdict
from io import StringIO
from typing import Any

from .models import TranscriptSegment


def build_feedback_csv(
    *,
    source_name: str,
    settings: dict[str, Any],
    audio_duration_seconds: float | None,
    processing_seconds: float | None,
    segments: list[TranscriptSegment],
    resource_history: list[dict],
    speaker_segments_count: int | None,
    output_json_path: str | None,
    include_text_samples: bool,
    local_diarization_error: str | None = None,
) -> str:
    rows: list[dict[str, str]] = []

    def add(category: str, metric: str, value: Any, unit: str = "", notes: str = "") -> None:
        rows.append(
            {
                "category": category,
                "metric": metric,
                "value": "" if value is None else str(value),
                "unit": unit,
                "notes": notes,
            }
        )

    speakers = sorted({segment.speaker for segment in segments})
    segment_durations = [
        max(0.0, (segment.end or 0.0) - (segment.start or 0.0))
        for segment in segments
        if segment.start is not None and segment.end is not None
    ]
    words = [word for segment in segments for word in segment.text.split()]
    unknown_segments = sum(1 for segment in segments if segment.speaker == "SPEAKER_UNKNOWN")

    add("run", "source_name", source_name)
    add("run", "status", "success")
    add("run", "output_json_path", output_json_path)
    add("audio", "duration", _round(audio_duration_seconds), "seconds")
    add("timing", "processing_time", _round(processing_seconds), "seconds")
    add("timing", "processing_to_audio_ratio", _ratio(processing_seconds, audio_duration_seconds), "x audio duration")

    for key, value in settings.items():
        redacted = "***" if "token" in key.lower() and value else value
        add("settings", key, redacted)

    add("transcript", "segment_count", len(segments))
    add("transcript", "speaker_count", len(speakers))
    add("transcript", "speakers", ", ".join(speakers))
    add("transcript", "unknown_speaker_segments", unknown_segments)
    add("transcript", "word_count", len(words))
    add("transcript", "character_count", sum(len(segment.text) for segment in segments))
    add("transcript", "avg_segment_duration", _avg(segment_durations), "seconds")
    add("transcript", "max_segment_duration", _max(segment_durations), "seconds")
    add("transcript", "segments_per_audio_hour", _per_hour(len(segments), audio_duration_seconds), "segments/hour")
    add("quality", "single_segment_output", len(segments) == 1)
    add("quality", "very_long_max_segment", bool(segment_durations and max(segment_durations) > 300))
    add("quality", "all_text_one_speaker", len(speakers) == 1 and len(segments) > 1)
    add("diarization", "speaker_time_segments", speaker_segments_count)
    if local_diarization_error:
        add("diarization", "error", local_diarization_error)

    for speaker, count in Counter(segment.speaker for segment in segments).items():
        add("speaker_segments", speaker, count, "segments")

    durations_by_speaker: dict[str, float] = defaultdict(float)
    for segment in segments:
        if segment.start is not None and segment.end is not None:
            durations_by_speaker[segment.speaker] += max(0.0, segment.end - segment.start)
    for speaker, duration in sorted(durations_by_speaker.items()):
        add("speaker_duration", speaker, _round(duration), "seconds")
    add(
        "speaker_balance",
        "dominant_speaker_share",
        dominant_speaker_share(durations_by_speaker),
        "share",
        "High values can be normal for interviews, but they help spot failed diarization.",
    )

    add_resource_summary(rows, resource_history)

    if include_text_samples:
        for label, segment in sample_segments(segments):
            add("text_sample", label, segment.text[:300], notes=f"{segment.timestamp} {segment.speaker}")
    else:
        add("privacy", "text_samples_included", False, notes="Enable in app if text snippets are needed for debugging.")

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["category", "metric", "value", "unit", "notes"])
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def add_resource_summary(rows: list[dict[str, str]], resource_history: list[dict]) -> None:
    def add(metric: str, value: Any, unit: str = "") -> None:
        rows.append({"category": "resources", "metric": metric, "value": str(value), "unit": unit, "notes": ""})

    if not resource_history:
        add("samples", 0)
        return

    add("samples", len(resource_history))
    for key in [
        "CPU gesamt %",
        "WhisperX CPU %",
        "RAM gesamt %",
        "Speicherdruck %",
        "WhisperX RAM GB",
        "RAM genutzt GB",
    ]:
        values = [float(item.get(key, 0.0) or 0.0) for item in resource_history]
        add(f"peak_{key}", _round(max(values)), "%" if "%" in key else "GB")
        add(f"avg_{key}", _avg(values), "%" if "%" in key else "GB")


def sample_segments(segments: list[TranscriptSegment]) -> list[tuple[str, TranscriptSegment]]:
    if not segments:
        return []
    indexes = sorted({0, len(segments) // 2, len(segments) - 1})
    labels = ["first", "middle", "last"]
    return [(labels[position], segments[index]) for position, index in enumerate(indexes)]


def _avg(values: list[float]) -> str:
    if not values:
        return ""
    return str(_round(sum(values) / len(values)))


def _max(values: list[float]) -> str:
    if not values:
        return ""
    return str(_round(max(values)))


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return _round(numerator / denominator)


def _per_hour(count: int, seconds: float | None) -> float | None:
    if seconds is None or seconds <= 0:
        return None
    return _round(count / (seconds / 3600))


def dominant_speaker_share(durations_by_speaker: dict[str, float]) -> float | None:
    total = sum(durations_by_speaker.values())
    if total <= 0:
        return None
    return _round(max(durations_by_speaker.values()) / total)
