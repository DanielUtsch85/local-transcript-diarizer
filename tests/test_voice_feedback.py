import csv
from io import StringIO

from voice_pipeline.feedback import build_voice_feedback_csv


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
        reference_metadata={"refs": [{"file": "ref_001.wav"}], "target_total_sec": 60, "actual_total_sec": 3.0},
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
    assert metrics[("synthesis", "error")] == "Install coqui-tts/TTS to use the XTTS backend"
