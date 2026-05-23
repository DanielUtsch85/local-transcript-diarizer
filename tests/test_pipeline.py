import csv
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from transcript_mvp.estimates import default_ratio, estimate_processing_seconds
from transcript_mvp.feedback import build_feedback_csv
from transcript_mvp.local_diarization import format_local_diarize_error, is_corrupt_silero_vad_error, run_local_diarize
from transcript_mvp.models import SpeakerMapping, SpeakerSegment, TranscriptSegment, format_timestamp
from transcript_mvp.pipeline import (
    assign_speakers_by_overlap,
    best_speaker_for_segment,
    build_whisperx_command,
    extract_speakers,
    format_whisperx_error,
    is_corrupt_alignment_cache_error,
    is_missing_model_cache_error,
    is_silero_download_error,
    is_torchvision_compatibility_error,
    parse_whisperx_progress,
    render_segments,
    run_whisperx,
    transcript_from_stdout,
)
from transcript_mvp.progress import ProgressReporter
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
            batch_size=1,
            chunk_size=10,
            threads=4,
            no_align=True,
            vad_method="pyannote",
        )

        self.assertIn("--batch_size", command)
        self.assertIn("1", command)
        self.assertIn("--chunk_size", command)
        self.assertIn("10", command)
        self.assertIn("--no_align", command)
        self.assertIn("--vad_method", command)
        self.assertIn("pyannote", command)
        self.assertIn("--print_progress", command)
        # WhisperX-Diarisierung ist bewusst deaktiviert.
        self.assertNotIn("--diarize", command)
        self.assertNotIn("--min_speakers", command)
        self.assertNotIn("--max_speakers", command)

    def test_run_whisperx_terminates_process_on_exception(self):
        process = Mock()
        process.stdout = iter([])
        process.poll.return_value = None
        process.returncode = 1
        process.pid = 1234

        reporter = ProgressReporter(
            on_overall=lambda value, text: None,
            on_transcription=lambda value: None,
            on_tick=lambda elapsed, pid: (_ for _ in ()).throw(RuntimeError("stop")),
            on_log=lambda line: None,
        )

        with tempfile.TemporaryDirectory() as directory, patch(
            "transcript_mvp.pipeline.subprocess.Popen",
            return_value=process,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                run_whisperx(
                    audio_path=Path("/tmp/audio.mp3"),
                    output_dir=Path(directory),
                    model="base",
                    language="de",
                    min_speakers=1,
                    max_speakers=2,
                    batch_size=1,
                    chunk_size=10,
                    threads=4,
                    no_align=True,
                    vad_method="pyannote",
                    reporter=reporter,
                )

        process.terminate.assert_called_once_with()

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

    def test_detects_torchvision_compatibility_error(self):
        details = (
            "RuntimeError: operator torchvision::nms does not exist "
            "ModuleNotFoundError: Could not import module 'Wav2Vec2ForCTC'."
        )

        self.assertTrue(is_torchvision_compatibility_error(details))
        self.assertIn("PyTorch-Pakete", format_whisperx_error(details))

    def test_detects_corrupt_alignment_cache_error(self):
        details = (
            "Downloading wav2vec2_fairseq_base_ls960_asr_ls960.pth "
            "RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory"
        )

        self.assertTrue(is_corrupt_alignment_cache_error(details))
        self.assertIn("wortgenauen Ausrichtung", format_whisperx_error(details))

    def test_detects_corrupt_alignment_cache_error_without_filename(self):
        details = (
            "File whisperx/alignment.py load_align_model "
            "File torchaudio/pipelines/_wav2vec2/impl.py "
            "RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory"
        )

        self.assertTrue(is_corrupt_alignment_cache_error(details))

    def test_detects_corrupt_silero_vad_error(self):
        details = "PytorchStreamReader failed reading zip archive: failed finding central directory"

        self.assertTrue(is_corrupt_silero_vad_error(details))
        self.assertIn("Silero-Modelldatei", format_local_diarize_error(details))

    def test_detects_silero_download_error(self):
        self.assertTrue(is_silero_download_error("torch.hub.load repo_or_dir='snakers4/silero-vad'"))
        self.assertTrue(is_silero_download_error("HTTP Error 403: rate limit exceeded"))
        self.assertFalse(is_silero_download_error("unrelated"))

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

    def test_assign_speakers_single_speaker_perfect_overlap(self):
        assigned = assign_speakers_by_overlap(
            [TranscriptSegment(start=0, end=5, speaker="SPEAKER_UNKNOWN", text="Hallo")],
            [SpeakerSegment(start=0, end=5, speaker="SPEAKER_00")],
        )

        self.assertEqual([segment.speaker for segment in assigned], ["SPEAKER_00"])

    def test_assign_speakers_two_speakers_non_overlapping(self):
        assigned = assign_speakers_by_overlap(
            [
                TranscriptSegment(start=0, end=4, speaker="SPEAKER_UNKNOWN", text="Hallo"),
                TranscriptSegment(start=4, end=8, speaker="SPEAKER_UNKNOWN", text="Guten Tag"),
            ],
            [
                SpeakerSegment(start=0, end=4, speaker="SPEAKER_00"),
                SpeakerSegment(start=4, end=8, speaker="SPEAKER_01"),
            ],
        )

        self.assertEqual([segment.speaker for segment in assigned], ["SPEAKER_00", "SPEAKER_01"])

    def test_assign_speakers_uses_greatest_overlap(self):
        assigned = assign_speakers_by_overlap(
            [TranscriptSegment(start=0, end=10, speaker="SPEAKER_UNKNOWN", text="Hallo")],
            [
                SpeakerSegment(start=0, end=3, speaker="SPEAKER_00"),
                SpeakerSegment(start=3, end=10, speaker="SPEAKER_01"),
            ],
        )

        self.assertEqual(assigned[0].speaker, "SPEAKER_01")

    def test_assign_speakers_keeps_original_labels_without_diarization(self):
        assigned = assign_speakers_by_overlap(
            [
                TranscriptSegment(start=0, end=4, speaker="ORIGINAL_00", text="Hallo"),
                TranscriptSegment(start=4, end=8, speaker="ORIGINAL_01", text="Guten Tag"),
            ],
            [],
            max_merged_duration=0,
        )

        self.assertEqual([segment.speaker for segment in assigned], ["ORIGINAL_00", "ORIGINAL_01"])

    def test_assign_speakers_handles_diarization_past_audio_end(self):
        assigned = assign_speakers_by_overlap(
            [TranscriptSegment(start=0, end=5, speaker="SPEAKER_UNKNOWN", text="Hallo")],
            [SpeakerSegment(start=0, end=30, speaker="SPEAKER_00")],
        )

        self.assertEqual(assigned[0].speaker, "SPEAKER_00")

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

    def test_progress_reporter_clamps_overall_progress(self):
        events = []
        reporter = ProgressReporter(
            on_overall=lambda value, text: events.append((value, text)),
            on_transcription=lambda value: None,
            on_tick=lambda elapsed, pid: None,
            on_log=lambda line: None,
        )

        reporter.overall(1.5, "zu viel")
        reporter.overall(-0.5, "zu wenig")

        self.assertEqual(events, [(1.0, "zu viel"), (0.0, "zu wenig")])

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
            local_diarization_error="Format not recognised",
            run_timestamp="2026-05-23T10:00:00",
            audio_filename="sample.wav",
            whisperx_command="whisperx sample.wav --model base",
        )
        rows = list(csv.DictReader(StringIO(csv_text)))
        metrics = {(row["category"], row["metric"]): row["value"] for row in rows}

        self.assertEqual(metrics[("run", "source_name")], "sample")
        self.assertEqual(metrics[("run", "status")], "partial")
        self.assertIn(("run", "timestamp"), metrics)
        self.assertIn(("run", "audio_filename"), metrics)
        self.assertEqual(metrics[("transcript", "segment_count")], "2")
        self.assertEqual(metrics[("transcript", "speaker_count")], "2")
        self.assertEqual(metrics[("transcript", "segments_per_audio_hour")], "720.0")
        self.assertEqual(metrics[("speaker_balance", "dominant_speaker_share")], "0.5")
        self.assertEqual(metrics[("resources", "samples")], "1")
        self.assertEqual(metrics[("quality", "single_segment_output")], "False")
        self.assertEqual(metrics[("diarization", "error")], "Format not recognised")
        self.assertEqual(metrics[("privacy", "text_samples_included")], "False")

    def test_build_feedback_csv_status_success_without_error(self):
        csv_text = build_feedback_csv(
            source_name="sample",
            settings={"model": "base", "speaker_backend": "disabled"},
            audio_duration_seconds=10,
            processing_seconds=20,
            segments=[
                TranscriptSegment(start=0, end=5, speaker="SPEAKER_UNKNOWN", text="Hallo Welt"),
            ],
            resource_history=[],
            speaker_segments_count=None,
            output_json_path="/tmp/out.json",
            include_text_samples=False,
        )
        rows = list(csv.DictReader(StringIO(csv_text)))
        metrics = {(row["category"], row["metric"]): row["value"] for row in rows}

        self.assertEqual(metrics[("run", "status")], "success")

    def test_run_local_diarize_converts_m4a_before_backend(self):
        captured = {}

        def fake_diarize(audio_path, **kwargs):
            captured["audio_path"] = Path(audio_path)
            captured["kwargs"] = kwargs
            return types.SimpleNamespace(
                segments=[
                    types.SimpleNamespace(start=0.0, end=1.5, speaker="SPEAKER_00"),
                    types.SimpleNamespace(start=1.5, end=3.0, speaker="SPEAKER_01"),
                ]
            )

        fake_module = types.SimpleNamespace(diarize=fake_diarize)
        completed = subprocess.CompletedProcess(args=["ffmpeg"], returncode=0)

        with patch.dict(sys.modules, {"diarize": fake_module}), patch(
            "transcript_mvp.local_diarization.subprocess.run",
            return_value=completed,
        ) as run:
            segments = run_local_diarize(Path("/tmp/Steffi-Daniel.m4a"), min_speakers=1, max_speakers=2)

        command = run.call_args.args[0]
        self.assertIn("ffmpeg", command[0])
        self.assertIn("-ar", command)
        self.assertIn("16000", command)
        self.assertEqual(captured["audio_path"].suffix, ".wav")
        self.assertEqual(captured["kwargs"]["min_speakers"], 1)
        self.assertEqual(captured["kwargs"]["max_speakers"], 2)
        self.assertEqual([segment.speaker for segment in segments], ["SPEAKER_00", "SPEAKER_01"])

    def test_run_local_diarize_converts_wav_to_16khz(self):
        captured = {}

        def fake_diarize(audio_path, **kwargs):
            captured["audio_path"] = Path(audio_path)
            captured["kwargs"] = kwargs
            return types.SimpleNamespace(
                segments=[
                    types.SimpleNamespace(start=0.0, end=1.5, speaker="SPEAKER_00"),
                    types.SimpleNamespace(start=1.5, end=3.0, speaker="SPEAKER_01"),
                ]
            )

        fake_module = types.SimpleNamespace(diarize=fake_diarize)
        completed = subprocess.CompletedProcess(args=["ffmpeg"], returncode=0)
        input_path = Path("/tmp/interview.wav")

        with patch.dict(sys.modules, {"diarize": fake_module}), patch(
            "transcript_mvp.local_diarization.subprocess.run",
            return_value=completed,
        ) as run:
            segments = run_local_diarize(input_path, min_speakers=1, max_speakers=2)

        command = run.call_args.args[0]
        self.assertIn("ffmpeg", command[0])
        self.assertIn("-ar", command)
        self.assertEqual(command[command.index("-ar") + 1], "16000")
        self.assertIn("-ac", command)
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(captured["audio_path"].suffix, ".wav")
        self.assertNotEqual(captured["audio_path"], input_path)
        self.assertEqual(captured["kwargs"]["min_speakers"], 1)
        self.assertEqual(captured["kwargs"]["max_speakers"], 2)
        self.assertEqual([segment.speaker for segment in segments], ["SPEAKER_00", "SPEAKER_01"])

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
                        "settings,vad_method,pyannote,,",
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
                vad_method="pyannote",
                history_dir=Path(directory),
            )

        self.assertEqual(estimate.samples, 1)
        self.assertEqual(estimate.seconds, 160)
        self.assertEqual(estimate.source, "exakt gleiche lokale Einstellungen")

    def test_default_ratio_is_conservative_for_cpu(self):
        # Werte muessen hoch genug sein, damit Nutzer nicht von echten Laufzeiten
        # ueberrascht werden. Groessenordnung: medium > 2x, large-v3 > 5x.
        self.assertGreater(
            default_ratio(model="medium", speaker_backend="diarize", memory_mode=True, no_align=True),
            2.0,
        )
        self.assertGreater(
            default_ratio(model="large-v3", speaker_backend="disabled", memory_mode=True, no_align=True),
            5.0,
        )
        # "Ohne Token" darf nicht mehr als gueltiger Backend-Key fungieren.
        ratio_without_legacy = default_ratio(
            model="medium",
            speaker_backend="Ohne Token",
            memory_mode=True,
            no_align=True,
        )
        ratio_disabled = default_ratio(
            model="medium",
            speaker_backend="disabled",
            memory_mode=True,
            no_align=True,
        )
        self.assertEqual(ratio_without_legacy, ratio_disabled)


if __name__ == "__main__":
    unittest.main()
