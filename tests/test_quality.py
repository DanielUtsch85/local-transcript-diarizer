import wave

import numpy as np

from voice_pipeline.audio_io import create_dummy_wav
from voice_pipeline.quality import score_segment


def _write_wav(path, samples, sample_rate=24000):
    pcm = np.clip(np.asarray(samples) * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def test_duration_is_measured(tmp_path):
    wav = create_dummy_wav(tmp_path / "sample.wav", duration_sec=2.0, sample_rate=24000, tone_hz=440)

    row = score_segment(wav, quality_config={"min_speech_ratio": 0.0, "min_rms_db": -80, "max_peak_db": 0})

    assert row["duration_sec"] == 2.0


def test_clipping_ratio_is_measured(tmp_path):
    samples = np.ones(24000, dtype=np.float32)
    wav = tmp_path / "clipped.wav"
    _write_wav(wav, samples)

    row = score_segment(wav, quality_config={"min_speech_ratio": 0.0, "min_rms_db": -80, "max_peak_db": 0})

    assert row["clipping_ratio"] > 0.99
    assert "too_much_clipping" in row["reject_reasons"]


def test_too_short_reject_reason(tmp_path):
    wav = create_dummy_wav(tmp_path / "short.wav", duration_sec=0.5, sample_rate=24000, tone_hz=440)

    row = score_segment(wav, quality_config={"min_duration_sec": 2.0, "min_speech_ratio": 0.0, "min_rms_db": -80, "max_peak_db": 0})

    assert row["accepted"] is False
    assert "too_short" in row["reject_reasons"]


def test_too_quiet_reject_reason(tmp_path):
    wav = create_dummy_wav(tmp_path / "quiet.wav", duration_sec=2.0, sample_rate=24000)

    row = score_segment(wav, quality_config={"min_speech_ratio": 0.0, "min_rms_db": -35, "max_peak_db": 0})

    assert "too_quiet" in row["reject_reasons"]
