import csv
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from transcript_mvp.estimates import estimate_processing_seconds
from transcript_mvp.feedback import build_feedback_csv
from transcript_mvp.models import SpeakerMapping, SpeakerSegment, TranscriptSegment, format_timestamp
from transcript_mvp.pipeline import (
    assign_speakers_by_overlap,
    best_speaker_for_segment,
    build_whisperx_command,
    extract_speakers,
    is_missing_model_cache_error,
    parse_whisperx_progress,
    render_segments,
    transcript_from_stdout,
)
from transcript_mvp.resources import _pressure_label


class PipelineTests(unittest.TestCase):
    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(3), "00:03")
        self.assertEqual(format_timestamp(65), "01:05")
        self.assertEqual(format_timestamp(3661), "01:01:01")

    def test_render_segments_merges_adjacent_speakers(self):
        transcript = {
            "segments": [
                {"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "Hallo"},
                {"start": 1, "end": 2, "speaker": "SPEAKER_00", "text": "zusammen"},
                {"start": 2, "end": 3, "speaker": "SPEAKER_01", "text": "Guten Tag"},
            ]
        }

        segments = render_segments(transcript)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "Hallo zusammen")
        self.assertEqual(segments[0].timestamp, "00:00")
        self.assertEqual(extract_speakers(segments), ["SPEAKER_00", "SPEAKER_01"])

    def test_render_segments_can_skip_merge(self):
        transcript = {
            "segments": [
                {"start": 0, "end": 1, "speaker": "SPEAKER_UNKNOWN", "text": "Hallo"},
                {"start": 1, "end": 2, "speaker": "SPEAKER_UNKNOWN", "text": "zusammen"},
            ]
        }

        segments = render_segments(transcript, merge_adjacent=False)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "Hallo")
        self.assertEqual(segments[1].text, "zusammen")

    def test_speaker_mapping_falls_back_to_original_label(self):
        mapping = SpeakerMapping({"SPEAKER_00": "Daniel"})

        self.assertEqual(mapping.label_for("SPEAKER_00"), "Daniel")
        self.assertEqual(mapping.label_for("SPEAKER_01"), "SPEAKER_01")

    def test_build_whisperx_command_includes_memory_options(self):
        command = build_whisperx_command(
            audio_path="/tmp/audio.mp3",
            output_dir="/tmp/out",
            model="base",
            language="de",
            min_speakers=1,
            max_speakers=2,
            batch_size=1,
            chunk_size=10,
            threads=4,
            no_align=True,
        )

        self.assertIn("--batch_size", command)
        self.assertIn("1", command)
        self.assertIn("--chunk_size", command)
        self.assertIn("10", command)
        self.assertIn("--no_align", command)
        self.assertIn("--vad_method", command)
        self.assertIn("silero", command)
        self.assertIn("--print_progress", command)
        self.assertNotIn("--diarize", command)

    def test_transcript_from_stdout(self):
        transcript = transcript_from_stdout(
            [
                "Progress: 99.83%...",
                "Transcript: [4655.911 --> 4664.618] Und jetzt sehen ich was.",
            ]
        )

        self.assertEqual(len(transcript["segments"]), 1)
        self.assertEqual(transcript["segments"][0]["speaker"], "SPEAKER_UNKNOWN")
        self.assertEqual(transcript["segments"][0]["start"], 4655.911)
        self.assertEqual(transcript["segments"][0]["text"], "Und jetzt sehen ich was.")

    def test_parse_whisperx_progress(self):
        self.assertEqual(parse_whisperx_progress("Progress: 98.81%..."), 0.9881)
        self.assertEqual(parse_whisperx_progress("Progress: 100.00%..."), 1.0)
        self.assertIsNone(parse_whisperx_progress("Transcript: text"))

    def test_detects_missing_model_cache_error(self):
        self.assertTrue(is_missing_model_cache_error("LocalEntryNotFoundError"))
        self.assertTrue(is_missing_model_cache_error("Failed to resolve 'huggingface.co'"))
        self.assertFalse(is_missing_model_cache_error("unrelated"))

    def test_assign_speakers_by_overlap(self):
        transcript_segments = [
            TranscriptSegment(start=0, end=5, speaker="SPEAKER_UNKNOWN", text="Hallo"),
            TranscriptSegment(start=5, end=10, speaker="SPEAKER_UNKNOWN", text="Guten Tag"),
        ]
        speaker_segments = [
            SpeakerSegment(start=0, end=4, speaker="SPEAKER_00"),
            SpeakerSegment(start=4, end=10, speaker="SPEAKER_01"),
        ]

        assigned = assign_speakers_by_overlap(transcript_segments, speaker_segments)

        self.assertEqual(assigned[0].speaker, "SPEAKER_00")
        self.assertEqual(assigned[1].speaker, "SPEAKER_01")

    def test_assign_speakers_does_not_create_huge_merged_segment(self):
        transcript_segments = [
            TranscriptSegment(start=0, end=80, speaker="SPEAKER_UNKNOWN", text="A"),
            TranscriptSegment(start=80, end=160, speaker="SPEAKER_UNKNOWN", text="B"),
        ]
        speaker_segments = [SpeakerSegment(start=0, end=160, speaker="SPEAKER_00")]

        assigned = assign_speakers_by_overlap(transcript_segments, speaker_segments, max_merged_duration=120)

        self.assertEqual(len(assigned), 2)

    def test_best_speaker_uses_nearest_when_no_overlap(self):
        speaker = best_speaker_for_segment(
            TranscriptSegment(start=10, end=11, speaker="SPEAKER_UNKNOWN", text="Hallo"),
            [SpeakerSegment(start=0, end=3, speaker="SPEAKER_00")],
        )

        self.assertEqual(speaker, "SPEAKER_00")

    def test_pressure_labels(self):
        self.assertEqual(_pressure_label(50), "niedrig")
        self.assertEqual(_pressure_label(75), "mittel")
        self.assertEqual(_pressure_label(88), "hoch")
        self.assertEqual(_pressure_label(95), "kritisch")

    def test_build_feedback_csv_contains_expected_metrics(self):
        csv_text = build_feedback_csv(
            source_name="sample",
            settings={"model": "base", "speaker_backend": "diarize"},
            audio_duration_seconds=10,
            processing_seconds=20,
            segments=[
                TranscriptSegment(start=0, end=5, speaker="SPEAKER_00", text="Hallo Welt"),
                TranscriptSegment(start=5, end=10, speaker="SPEAKER_01", text="Guten Tag"),
            ],
            resource_history=[
                {
                    "Zeit": 1,
                    "CPU gesamt %": 10,
                    "WhisperX CPU %": 20,
                    "RAM gesamt %": 30,
                    "Speicherdruck %": 40,
                    "WhisperX RAM GB": 1.2,
                    "RAM genutzt GB": 8.0,
                }
            ],
            speaker_segments_count=4,
            output_json_path="/tmp/out.json",
            include_text_samples=False,
        )
        rows = list(csv.DictReader(StringIO(csv_text)))
        metrics = {(row["category"], row["metric"]): row["value"] for row in rows}

        self.assertEqual(metrics[("run", "source_name")], "sample")
        self.assertEqual(metrics[("transcript", "segment_count")], "2")
        self.assertEqual(metrics[("transcript", "speaker_count")], "2")
        self.assertEqual(metrics[("transcript", "segments_per_audio_hour")], "720.0")
        self.assertEqual(metrics[("speaker_balance", "dominant_speaker_share")], "0.5")
        self.assertEqual(metrics[("resources", "samples")], "1")
        self.assertEqual(metrics[("quality", "single_segment_output")], "False")
        self.assertEqual(metrics[("privacy", "text_samples_included")], "False")

    def test_estimate_processing_uses_feedback_history(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run-1"
            run_dir.mkdir()
            (run_dir / "feedback.csv").write_text(
                "\n".join(
                    [
                        "category,metric,value,unit,notes",
                        "audio,duration,100,seconds,",
                        "timing,processing_time,80,seconds,",
                        "settings,model,medium,,",
                        "settings,speaker_backend,diarize,,",
                        "settings,memory_mode,True,,",
                        "settings,no_align,True,,",
                        "settings,batch_size,1,,",
                        "settings,chunk_size,10,,",
                        "settings,threads,4,,",
                    ]
                ),
                encoding="utf-8",
            )

            estimate = estimate_processing_seconds(
                audio_seconds=200,
                model="medium",
                speaker_backend="diarize",
                memory_mode=True,
                no_align=True,
                batch_size=1,
                chunk_size=10,
                threads=4,
                history_dir=Path(directory),
            )

        self.assertEqual(estimate.samples, 1)
        self.assertEqual(estimate.seconds, 160)
        self.assertEqual(estimate.source, "exakt gleiche lokale Einstellungen")


if __name__ == "__main__":
    unittest.main()
