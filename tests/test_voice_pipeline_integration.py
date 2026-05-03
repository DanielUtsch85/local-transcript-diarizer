import json
import shutil

from voice_pipeline.audio_io import create_dummy_wav, cut_segment
from voice_pipeline.logging_utils import read_jsonl, write_jsonl
from voice_pipeline.quality import score_segment
from voice_pipeline.segment_loader import filter_segments, load_segments, segment_filename


def test_minimal_curation_flow(tmp_path):
    if shutil.which("ffmpeg") is None:
        return
    source = create_dummy_wav(tmp_path / "interview.wav", duration_sec=4.0, sample_rate=24000, tone_hz=440)
    diarization = tmp_path / "diarization.json"
    diarization.write_text(
        json.dumps([{"speaker": "SPEAKER_02", "start": 0.2, "end": 3.2, "source_file": str(source)}]),
        encoding="utf-8",
    )
    raw_dir = tmp_path / "output" / "SPEAKER_02" / "raw_segments"
    manifest = tmp_path / "output" / "SPEAKER_02" / "manifests" / "segments.jsonl"

    selected = filter_segments(load_segments(diarization), "SPEAKER_02")
    dest = raw_dir / segment_filename(1, selected[0])
    cut_segment(selected[0].source_file, dest, selected[0].start, selected[0].end, padding_ms=0)
    row = score_segment(dest, speaker="SPEAKER_02", quality_config={"min_speech_ratio": 0.0, "min_rms_db": -80, "max_peak_db": 0})
    write_jsonl(manifest, [row])

    rows = read_jsonl(manifest)

    assert dest.exists()
    assert rows[0]["speaker"] == "SPEAKER_02"
    assert "accepted" in rows[0]
