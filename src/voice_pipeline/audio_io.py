from __future__ import annotations

import math
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .logging_utils import write_json


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")


def wav_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
    return {
        "file": str(path),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "duration_sec": frames / sample_rate if sample_rate else 0.0,
    }


def prepare_audio(input_path: Path, output_path: Path, sample_rate: int = 24000, mono: bool = True) -> dict[str, Any]:
    if sample_rate not in {16000, 24000}:
        raise ValueError("sample_rate must be 16000 or 24000")
    require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1" if mono else "2",
        "-ar",
        str(sample_rate),
        "-af",
        "loudnorm=I=-23:LRA=7:TP=-2",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = wav_info(output_path)
    metadata = {
        "source_file": str(input_path),
        "prepared_file": str(output_path),
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "duration_sec": info["duration_sec"],
    }
    write_json(output_path.with_suffix(output_path.suffix + ".metadata.json"), metadata)
    return metadata


def cut_segment(
    source_file: Path,
    output_file: Path,
    start_sec: float,
    end_sec: float,
    padding_ms: int = 100,
) -> Path:
    require_ffmpeg()
    padded_start = max(0.0, start_sec - padding_ms / 1000)
    padded_end = max(padded_start, end_sec + padding_ms / 1000)
    duration = padded_end - padded_start
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{padded_start:.3f}",
        "-i",
        str(source_file),
        "-t",
        f"{duration:.3f}",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_file),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_file


def denoise_wav(input_path: Path, output_path: Path, noise_reduction_db: float = 12.0) -> Path:
    """Apply a conservative ffmpeg frequency-domain denoise filter to a WAV file."""
    require_ffmpeg()
    noise_reduction_db = max(0.0, min(float(noise_reduction_db), 30.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        f"afftdn=nr={noise_reduction_db:.1f}",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported for scoring: {path}")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def create_dummy_wav(path: Path, duration_sec: float = 1.0, sample_rate: int = 24000, tone_hz: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(duration_sec * sample_rate)
    if tone_hz:
        t = np.arange(frame_count, dtype=np.float32) / sample_rate
        samples = 0.15 * np.sin(2 * math.pi * tone_hz * t)
    else:
        samples = np.zeros(frame_count, dtype=np.float32)
    pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path
