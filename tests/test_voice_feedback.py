import csv
from io import StringIO

from voice_pipeline.feedback import build_synthesis_review_csv, build_voice_feedback_csv


def test_voice_feedback_csv_contains_quality_metrics(tmp_path):
    csv_text = build_voice_feedback_csv(
        speaker="SPEAKER_00",
        source_audio=tmp_path / "audio.wav",
        diarization_path=tmp_path / "diarization.json",
        settings={"reject_overlaps": True, "sample_rate": 24000},
        raw_rows=[{"duration_sec": 3.0}, {"duration_sec": 4.0}],
        scored_rows=[
            {"duration_sec": 3.0, "accepted": True, "overlap_sec": 0.0},
            {"duration_sec": 4.0, "accepted": False, "overlap_sec": 0.5, "reject_reasons": ["overlaps_other_speaker"]},
        ],
        accepted_rows=[{"duration_sec": 3.0, "accepted": True}],
        rejected_rows=[{"duration_sec": 4.0, "accepted": False, "reject_reasons": ["overlaps_other_speaker"]}],
        reference_metadata={
            "refs": [{"file": "ref_001.wav"}],
            "target_total_sec": 60,
            "actual_total_sec": 3.0,
            "selection_mode": "automatic",
            "denoise_enabled": True,
            "denoise_strength_db": 12.0,
        },
        synthesis_error="Install coqui-tts/TTS to use the XTTS backend",
    )

    rows = list(csv.DictReader(StringIO(csv_text)))
    metrics = {(row["category"], row["metric"]): row["value"] for row in rows}

    assert metrics[("run", "speaker")] == "SPEAKER_00"
    assert metrics[("segments", "accepted_count")] == "1"
    assert metrics[("segments", "acceptance_rate")] == "0.5"
    assert metrics[("overlap", "rejected_for_overlap")] == "1"
    assert metrics[("reject_reason", "overlaps_other_speaker")] == "1"
    assert metrics[("reference_pack", "ref_count")] == "1"
    assert metrics[("reference_pack", "denoise_enabled")] == "True"
    assert metrics[("reference_pack", "denoise_strength_db")] == "12.0"
    assert metrics[("synthesis", "error")] == "Install coqui-tts/TTS to use the XTTS backend"


def test_synthesis_review_csv_combines_technical_and_human_metrics(tmp_path):
    technical = build_voice_feedback_csv(
        speaker="SPEAKER_00",
        source_audio=tmp_path / "audio.wav",
        diarization_path=tmp_path / "diarization.json",
        settings={"backend": "xtts", "sample_rate": 24000},
        raw_rows=[{"duration_sec": 3.0}],
        scored_rows=[{"duration_sec": 3.0, "accepted": True, "overlap_sec": 0.0}],
        accepted_rows=[{"duration_sec": 3.0, "accepted": True}],
        rejected_rows=[],
        reference_metadata={"refs": [{"file": "ref_001.wav"}], "target_total_sec": 60, "actual_total_sec": 3.0},
        synthesis_metadata={"backend": "xtts", "synthetic": True, "output_file": "sample.wav"},
    )

    review_csv = build_synthesis_review_csv(
        speaker="SPEAKER_00",
        sample_wav=tmp_path / "sample.wav",
        technical_feedback_csv=technical,
        review={"overall_quality_1_to_5": 4, "notes": "nah dran"},
    )

    rows = list(csv.DictReader(StringIO(review_csv)))
    metrics = {(row["category"], row["metric"]): row["value"] for row in rows}

    assert metrics[("technical_settings", "backend")] == "xtts"
    assert metrics[("technical_reference_pack", "ref_count")] == "1"
    assert metrics[("human_review", "overall_quality_1_to_5")] == "4"
    assert metrics[("human_review", "notes")] == "nah dran"


def test_synthesis_review_csv_uses_current_sample_sidecar_metadata(tmp_path):
    technical = build_voice_feedback_csv(
        speaker="SPEAKER_00",
        source_audio=tmp_path / "audio.wav",
        diarization_path=tmp_path / "diarization.json",
        settings={"backend": "xtts"},
        raw_rows=[],
        scored_rows=[],
        accepted_rows=[],
        rejected_rows=[],
        synthesis_metadata={"backend": "xtts", "synthetic": True, "output_file": "old_sample.wav"},
    )

    review_csv = build_synthesis_review_csv(
        speaker="SPEAKER_00",
        sample_wav=tmp_path / "new_sample.wav",
        technical_feedback_csv=technical,
        review={"overall_quality_1_to_5": 2},
        sample_synthesis_metadata={
            "backend": "xtts",
            "synthetic": True,
            "output_file": "new_sample.wav",
            "reference_files": ["ref_001.wav", "ref_002.wav"],
        },
    )

    rows = list(csv.DictReader(StringIO(review_csv)))
    metrics = {(row["category"], row["metric"]): row["value"] for row in rows}

    assert metrics[("technical_synthesis", "output_file")] == "new_sample.wav"
    assert metrics[("technical_synthesis", "reference_file_count")] == "2"
