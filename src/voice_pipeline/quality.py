from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .audio_io import read_wav_mono
from .segment_loader import DiarizationSegment
from .vad import speech_ratio

SOFT_SPEECH_REJECT_REASONS = {"low_speech_ratio", "too_silent"}


def _db(value: float) -> float:
    if value <= 0:
        return -120.0
    return 20.0 * math.log10(value)


def score_segment(
    path: Path,
    speaker: str | None = None,
    quality_config: dict[str, Any] | None = None,
    transcript: str | None = None,
    language: str | None = None,
    transcript_confidence: float | None = None,
    overlap_sec: float = 0.0,
    overlap_speakers: list[str] | None = None,
) -> dict[str, Any]:
    cfg = quality_config or {}
    try:
        samples, sample_rate = read_wav_mono(path)
        duration_sec = float(samples.size / sample_rate) if sample_rate else 0.0
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        clipping_ratio = float(np.mean(np.abs(samples) >= 0.999)) if samples.size else 0.0
        vad = speech_ratio(path)
        silence_ratio = max(0.0, 1.0 - vad.speech_ratio)
        zcr = float(np.mean(np.abs(np.diff(np.signbit(samples))))) if samples.size > 1 else 0.0
        quiet = np.abs(samples) < max(rms * 0.5, 1e-5)
        noise_floor = float(np.sqrt(np.mean(np.square(samples[quiet])))) if np.any(quiet) else 1e-6
        snr = _db(rms / max(noise_floor, 1e-6))
        row: dict[str, Any] = {
            "file": str(path),
            "speaker": speaker,
            "duration_sec": round(duration_sec, 3),
            "rms_db": round(_db(rms), 3),
            "peak_db": round(_db(peak), 3),
            "clipping_ratio": round(clipping_ratio, 6),
            "speech_ratio": round(vad.speech_ratio, 6),
            "silence_ratio": round(silence_ratio, 6),
            "snr_estimate_simple": round(snr, 3),
            "zero_crossing_rate": round(zcr, 6),
            "vad_method": vad.method,
            "transcript": transcript,
            "language": language,
            "transcript_confidence": transcript_confidence,
            "overlap_sec": round(max(0.0, overlap_sec), 3),
            "overlap_ratio": 0.0,
            "overlap_speakers": sorted(set(overlap_speakers or [])),
        }
        row["overlap_ratio"] = round(row["overlap_sec"] / duration_sec, 6) if duration_sec else 0.0
    except Exception as exc:
        return {
            "file": str(path),
            "speaker": speaker,
            "accepted": False,
            "reject_reasons": ["decode_error"],
            "error": str(exc),
        }

    reasons: list[str] = []
    if row["duration_sec"] < float(cfg.get("min_duration_sec", 2.0)):
        reasons.append("too_short")
    if row["duration_sec"] > float(cfg.get("max_duration_sec", 15.0)):
        reasons.append("too_long")
    if row["speech_ratio"] < float(cfg.get("min_speech_ratio", 0.75)):
        reasons.append("low_speech_ratio")
        reasons.append("too_silent")
    if row["clipping_ratio"] > float(cfg.get("max_clipping_ratio", 0.001)):
        reasons.append("too_much_clipping")
    if row["rms_db"] < float(cfg.get("min_rms_db", -35)):
        reasons.append("too_quiet")
    if row["peak_db"] > float(cfg.get("max_peak_db", -0.5)):
        reasons.append("too_much_clipping")
    if cfg.get("reject_empty_transcript", True) and transcript is not None and not transcript.strip():
        reasons.append("empty_transcript")
    if row["overlap_sec"] > float(cfg.get("max_overlap_sec", 0.0)):
        reasons.append("overlaps_other_speaker")

    row["accepted"] = not reasons
    row["reject_reasons"] = sorted(set(reasons))
    return row


def relax_speech_ratio_rejections(
    rows: list[dict[str, Any]],
    *,
    fallback_min_speech_ratio: float = 0.45,
    configured_min_speech_ratio: float | None = None,
) -> list[dict[str, Any]]:
    """Avoid an empty reference pack when VAD is stricter than the audio warrants."""
    if any(row.get("accepted") for row in rows):
        return list(rows)

    relaxed_rows: list[dict[str, Any]] = []
    for row in rows:
        reasons = set(row.get("reject_reasons") or [])
        speech_ratio_value = float(row.get("speech_ratio") or 0.0)
        if reasons and reasons.issubset(SOFT_SPEECH_REJECT_REASONS) and speech_ratio_value >= fallback_min_speech_ratio:
            relaxed = dict(row)
            relaxed["accepted"] = True
            relaxed["reject_reasons"] = []
            relaxed["quality_warnings"] = sorted(set(relaxed.get("quality_warnings") or []) | reasons)
            relaxed["acceptance_mode"] = "relaxed_speech_ratio"
            relaxed["fallback_min_speech_ratio"] = fallback_min_speech_ratio
            if configured_min_speech_ratio is not None:
                relaxed["configured_min_speech_ratio"] = configured_min_speech_ratio
            relaxed_rows.append(relaxed)
        else:
            relaxed_rows.append(row)
    return relaxed_rows


def segment_overlap(segment: DiarizationSegment, all_segments: list[DiarizationSegment]) -> dict[str, Any]:
    total = 0.0
    speakers: set[str] = set()
    for other in all_segments:
        if other.speaker == segment.speaker:
            continue
        if other.source_file != segment.source_file:
            continue
        overlap = min(segment.end, other.end) - max(segment.start, other.start)
        if overlap > 0:
            total += overlap
            speakers.add(other.speaker)
    return {"overlap_sec": round(total, 3), "overlap_speakers": sorted(speakers)}
