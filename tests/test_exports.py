import json

from transcript_mvp.exports import build_diarization_json, build_diarization_jsonl
from transcript_mvp.models import SpeakerMapping, SpeakerSegment, TranscriptSegment


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


def test_build_diarization_json_accepts_raw_speaker_segments():
    payload = build_diarization_json(
        [
            SpeakerSegment(start=0.1, end=1.5, speaker="SPEAKER_00"),
            SpeakerSegment(start=1.6, end=2.5, speaker="SPEAKER_01"),
        ],
        "/tmp/interview.wav",
    )

    rows = json.loads(payload)

    assert len(rows) == 2
    assert rows[0]["speaker"] == "SPEAKER_00"
    assert rows[0]["start"] == 0.1


def test_build_diarization_json_applies_speaker_mapping():
    payload = build_diarization_json(
        [
            SpeakerSegment(start=0.1, end=1.5, speaker="SPEAKER_UNKNOWN"),
        ],
        "/tmp/interview.wav",
        SpeakerMapping({"SPEAKER_UNKNOWN": "daniel"}),
    )

    rows = json.loads(payload)

    assert rows[0]["speaker"] == "daniel"


def test_build_diarization_jsonl_applies_speaker_mapping():
    payload = build_diarization_jsonl(
        [
            TranscriptSegment(start=0.1, end=1.5, speaker="SPEAKER_UNKNOWN", text="Hallo"),
        ],
        "/tmp/interview.wav",
        SpeakerMapping({"SPEAKER_UNKNOWN": "daniel"}),
    )

    rows = [json.loads(line) for line in payload.splitlines()]

    assert rows[0]["speaker"] == "daniel"
