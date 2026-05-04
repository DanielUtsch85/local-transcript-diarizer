from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples
    duration = samples.size / source_rate
    target_size = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


@lru_cache(maxsize=1)
def _load_silero_model():
    from silero_vad import load_silero_vad  # type: ignore

    return load_silero_vad()


def speech_ratio(path: Path) -> VadResult:
    samples, sample_rate = read_wav_mono(path)
    try:
        from silero_vad import get_speech_timestamps  # type: ignore

        model = _load_silero_model()
        vad_sample_rate = sample_rate if sample_rate in {8000, 16000} else 16000
        vad_samples = _resample_linear(samples, sample_rate, vad_sample_rate)
        timestamps = get_speech_timestamps(vad_samples, model, sampling_rate=vad_sample_rate)
        speech_samples = sum(max(0, item["end"] - item["start"]) for item in timestamps)
        return VadResult(speech_ratio=float(speech_samples / vad_samples.size) if vad_samples.size else 0.0, method="silero")
    except Exception:
        return VadResult(speech_ratio=_energy_speech_ratio(samples, sample_rate), method="energy")
