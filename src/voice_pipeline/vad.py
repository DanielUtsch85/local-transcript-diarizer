from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_io import read_wav_mono


@dataclass
class VadResult:
    speech_ratio: float
    method: str


def _energy_speech_ratio(samples: np.ndarray, sample_rate: int) -> float:
    if samples.size == 0:
        return 0.0
    frame_len = max(1, int(sample_rate * 0.03))
    frame_count = max(1, int(np.ceil(samples.size / frame_len)))
    padded = np.pad(samples, (0, frame_count * frame_len - samples.size))
    frames = padded.reshape(frame_count, frame_len)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    threshold = max(0.008, float(np.percentile(rms, 25) * 2.5))
    return float(np.mean(rms > threshold))


def speech_ratio(path: Path) -> VadResult:
    samples, sample_rate = read_wav_mono(path)
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad  # type: ignore

        model = load_silero_vad()
        timestamps = get_speech_timestamps(samples, model, sampling_rate=sample_rate)
        speech_samples = sum(max(0, item["end"] - item["start"]) for item in timestamps)
        return VadResult(speech_ratio=float(speech_samples / samples.size) if samples.size else 0.0, method="silero")
    except Exception:
        return VadResult(speech_ratio=_energy_speech_ratio(samples, sample_rate), method="energy")
