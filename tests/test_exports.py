import json

from transcript_mvp.exports import build_diarization_json, build_diarization_jsonl
from transcript_mvp.models import TranscriptSegment


def test_build_diarization_json_matches_voice_pipeline_schema():
    payload = build_diarization_json(
        [
            TranscriptSegment(start=1.25, end=4.5, speaker="SPEAKER_02", text="Hallo"),
            TranscriptSegment(start=None, end=None, speaker="SPEAKER_UNKNOWN", text="ohne Zeit"),
        ],
        "/tmp/interview.wav",
    )

    rows = json.loads(payload)

    assert rows == [
        {
            "speaker": "SPEAKER_02",
            "start": 1.25,
            "end": 4.5,
            "source_file": "/tmp/interview.wav",
        }
    ]


def test_build_diarization_jsonl_writes_one_object_per_line():
    payload = build_diarization_jsonl(
        [
            TranscriptSegment(start=1.0, end=2.0, speaker="SPEAKER_01", text="A"),
            TranscriptSegment(start=2.0, end=3.0, speaker="SPEAKER_02", text="B"),
        ],
        "/tmp/interview.wav",
    )

    lines = [json.loads(line) for line in payload.splitlines()]

    assert [row["speaker"] for row in lines] == ["SPEAKER_01", "SPEAKER_02"]
    assert all(row["source_file"] == "/tmp/interview.wav" for row in lines)
