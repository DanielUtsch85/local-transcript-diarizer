import json

import pytest

from voice_pipeline.segment_loader import filter_segments, load_segments, segment_filename


def test_load_segments_from_json(tmp_path):
    source = tmp_path / "audio.wav"
    path = tmp_path / "diarization.json"
    path.write_text(
        json.dumps(
            [
                {"speaker": "SPEAKER_02", "start": 1.0, "end": 5.5, "source_file": str(source)},
                {"speaker": "SPEAKER_01", "start": 6.0, "end": 7.0, "source_file": str(source)},
            ]
        ),
        encoding="utf-8",
    )

    segments = load_segments(path)

    assert len(segments) == 2
    assert segments[0].speaker == "SPEAKER_02"
    assert segments[0].duration_sec == 4.5


def test_load_segments_from_jsonl_and_filter(tmp_path):
    path = tmp_path / "diarization.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"speaker": "SPEAKER_02", "start": 1.0, "end": 5.0, "source_file": "audio.wav"}),
                json.dumps({"speaker": "SPEAKER_01", "start": 5.0, "end": 9.0, "source_file": "audio.wav"}),
            ]
        ),
        encoding="utf-8",
    )

    selected = filter_segments(load_segments(path), "SPEAKER_02")

    assert len(selected) == 1
    assert selected[0].source_file == tmp_path / "audio.wav"
    assert segment_filename(1, selected[0]) == "seg_000001_0001.000_0005.000.wav"


def test_invalid_segments_are_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"speaker": "SPEAKER_02", "start": 5.0, "end": 1.0, "source_file": "x.wav"}]), encoding="utf-8")

    with pytest.raises(ValueError):
        load_segments(path)
