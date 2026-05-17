import wave

import numpy as np

from voice_pipeline.audio_io import create_dummy_wav
from voice_pipeline.quality import relax_speech_ratio_rejections, score_segment, segment_overlap
from voice_pipeline.segment_loader import DiarizationSegment
from voice_pipeline.vad import _resample_linear


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


def test_overlap_reject_reason(tmp_path):
    wav = create_dummy_wav(tmp_path / "sample.wav", duration_sec=3.0, sample_rate=24000, tone_hz=440)

    row = score_segment(
        wav,
        quality_config={"min_speech_ratio": 0.0, "min_rms_db": -80, "max_peak_db": 0, "max_overlap_sec": 0.0},
        overlap_sec=0.4,
        overlap_speakers=["SPEAKER_01"],
    )

    assert row["overlap_sec"] == 0.4
    assert row["overlap_speakers"] == ["SPEAKER_01"]
    assert "overlaps_other_speaker" in row["reject_reasons"]


def test_relaxes_speech_ratio_only_when_pack_would_be_empty():
    rows = [
        {"accepted": False, "speech_ratio": 0.52, "reject_reasons": ["low_speech_ratio", "too_silent"]},
        {"accepted": False, "speech_ratio": 0.65, "reject_reasons": ["too_long", "low_speech_ratio"]},
        {"accepted": False, "speech_ratio": 0.35, "reject_reasons": ["low_speech_ratio"]},
    ]

    relaxed = relax_speech_ratio_rejections(rows, fallback_min_speech_ratio=0.45, configured_min_speech_ratio=0.75)

    assert relaxed[0]["accepted"] is True
    assert relaxed[0]["acceptance_mode"] == "relaxed_speech_ratio"
    assert relaxed[0]["quality_warnings"] == ["low_speech_ratio", "too_silent"]
    assert relaxed[0]["configured_min_speech_ratio"] == 0.75
    assert relaxed[1]["accepted"] is False
    assert relaxed[2]["accepted"] is False


def test_does_not_relax_when_any_segment_is_already_accepted():
    rows = [
        {"accepted": True, "speech_ratio": 0.8, "reject_reasons": []},
        {"accepted": False, "speech_ratio": 0.5, "reject_reasons": ["low_speech_ratio"]},
    ]

    relaxed = relax_speech_ratio_rejections(rows, fallback_min_speech_ratio=0.45)

    assert relaxed[1]["accepted"] is False


def test_segment_overlap_ignores_same_speaker_and_different_file(tmp_path):
    source = tmp_path / "audio.wav"
    other_source = tmp_path / "other.wav"
    segment = DiarizationSegment("SPEAKER_00", 10.0, 15.0, source)
    all_segments = [
        segment,
        DiarizationSegment("SPEAKER_01", 14.5, 16.0, source),
        DiarizationSegment("SPEAKER_00", 12.0, 13.0, source),
        DiarizationSegment("SPEAKER_02", 12.0, 13.0, other_source),
    ]

    overlap = segment_overlap(segment, all_segments)

    assert overlap == {"overlap_sec": 0.5, "overlap_speakers": ["SPEAKER_01"]}


def test_vad_resampler_maps_24khz_to_16khz_length():
    samples = np.linspace(-1.0, 1.0, 24000, dtype=np.float32)

    resampled = _resample_linear(samples, 24000, 16000)

    assert resampled.shape == (16000,)
    assert resampled.dtype == np.float32
