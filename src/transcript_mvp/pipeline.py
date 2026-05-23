from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO, Callable

from .models import SpeakerSegment, TranscriptSegment
from .progress import ProgressReporter


def create_run_dir(base_dir: Path, source_name: str) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", source_name).strip("-") or "audio"
    run_dir = base_dir / f"{safe_name}-{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_upload(uploaded_file: BinaryIO, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = re.sub(r"[^a-zA-Z0-9._-]+", "-", uploaded_file.name)
    target = upload_dir / filename
    target.write_bytes(uploaded_file.getvalue())
    return target


def run_whisperx(
    audio_path: Path,
    output_dir: Path,
    model: str,
    language: str | None,
    batch_size: int,
    chunk_size: int,
    threads: int,
    no_align: bool,
    vad_method: str,
    on_output: Callable[[str], None] | None = None,
    on_tick: Callable[[float, int], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
    reporter: ProgressReporter | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_whisperx_command(
        audio_path=audio_path,
        output_dir=output_dir,
        model=model,
        language=language,
        batch_size=batch_size,
        chunk_size=chunk_size,
        threads=threads,
        no_align=no_align,
        vad_method=vad_method,
    )

    started_at = time.monotonic()
    output_lines: list[str] = []
    line_queue: queue.Queue[str] = queue.Queue()
    log_path = output_dir / "whisperx.log"

    try:
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "WhisperX ist nicht installiert oder nicht im PATH. "
            "Bitte die Installation aus der README ausfuehren."
        ) from exc

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            line_queue.put(line.rstrip())

    reader = threading.Thread(target=read_output, daemon=True)

    def handle_output_line(line: str) -> None:
        output_lines.append(line)
        progress = parse_whisperx_progress(line)
        if progress is not None:
            if reporter:
                reporter.transcription(progress)
            elif on_progress:
                on_progress(progress)
        if reporter:
            reporter.log(line)
        elif on_output:
            on_output(line)

    def drain_output_queue() -> None:
        while not line_queue.empty():
            line = line_queue.get()
            if line:
                handle_output_line(line)

    try:
        reader.start()
        while process.poll() is None or not line_queue.empty():
            drain_output_queue()
            if reporter:
                reporter.tick(time.monotonic() - started_at, process.pid)
            elif on_tick:
                on_tick(time.monotonic() - started_at, process.pid)
            time.sleep(0.5)

        reader.join(timeout=1)
        drain_output_queue()

        if process.returncode != 0:
            details = "\n".join(output_lines[-80:]).strip()
            log_path.write_text("\n".join(output_lines), encoding="utf-8")
            raise RuntimeError(format_whisperx_error(details))

        log_path.write_text("\n".join(output_lines), encoding="utf-8")

        json_files = sorted(output_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not json_files:
            raise FileNotFoundError("WhisperX hat keine JSON-Ausgabe erzeugt.")
        return json_files[0]
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        reader.join(timeout=1)
        drain_output_queue()


# Sprechererkennung erfolgt separat ueber run_local_diarize().
# WhisperX wird ausschliesslich fuer Transkription und VAD verwendet:
# kein --diarize-Flag und keine Speaker-Constraints noetig.
def build_whisperx_command(
    audio_path: Path,
    output_dir: Path,
    model: str,
    language: str | None,
    batch_size: int,
    chunk_size: int,
    threads: int,
    no_align: bool,
    vad_method: str,
) -> list[str]:
    cmd = [
        "whisperx",
        str(audio_path),
        "--model",
        model,
        "--device",
        "cpu",
        "--compute_type",
        "int8",
        "--batch_size",
        str(batch_size),
        "--chunk_size",
        str(chunk_size),
        "--threads",
        str(threads),
        "--vad_method",
        vad_method,
        "--print_progress",
        "True",
        "--output_format",
        "json",
        "--output_dir",
        str(output_dir),
    ]

    if language:
        cmd.extend(["--language", language])
    if no_align:
        cmd.append("--no_align")
    return cmd


def transcript_from_stdout(output_lines: list[str]) -> dict[str, Any]:
    segments = []
    pattern = re.compile(r"^Transcript:\s*\[(?P<start>[\d.]+)\s*-->\s*(?P<end>[\d.]+)\]\s*(?P<text>.*)$")
    for line in output_lines:
        match = pattern.match(line.strip())
        if not match:
            continue
        text = match.group("text").strip()
        if not text:
            continue
        segments.append(
            {
                "start": float(match.group("start")),
                "end": float(match.group("end")),
                "speaker": "SPEAKER_UNKNOWN",
                "text": text,
            }
        )
    return {"segments": segments}


def parse_whisperx_progress(line: str) -> float | None:
    match = re.search(r"Progress:\s*(?P<percent>\d+(?:\.\d+)?)%", line)
    if not match:
        return None
    value = float(match.group("percent")) / 100.0
    return min(1.0, max(0.0, value))


def format_whisperx_error(details: str) -> str:
    if is_torchvision_compatibility_error(details):
        return (
            "WhisperX konnte nicht starten, weil die installierten PyTorch-Pakete nicht zusammenpassen.\n\n"
            "Bitte `torch`, `torchaudio` und `torchvision` in kompatiblen Versionen neu installieren. "
            "Fuer diese Umgebung wurde `torch==2.8.0`, `torchaudio==2.8.0` und `torchvision==0.23.0` getestet.\n\n"
            f"{details}"
        )
    if is_corrupt_alignment_cache_error(details):
        return (
            "WhisperX konnte die Datei transkribieren, ist aber bei der wortgenauen Ausrichtung gescheitert.\n\n"
            "Das englische Alignment-Modell wurde offenbar nur unvollstaendig in den Torch-Cache geladen. "
            "Bitte die defekte Cache-Datei loeschen und den Lauf erneut starten, oder in der App "
            "`Wortgenaue Ausrichtung sparen` aktivieren.\n\n"
            "Cache-Datei:\n"
            "`~/.cache/torch/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth`\n\n"
            f"{details}"
        )
    if is_silero_download_error(details):
        return (
            "WhisperX konnte die Datei nicht verarbeiten.\n\n"
            "Silero VAD ist nicht lokal zwischengespeichert und konnte wegen GitHub-Rate-Limit "
            "nicht heruntergeladen werden. Bitte VAD-Methode `pyannote` nutzen oder Silero einmalig "
            "mit Internetzugang vorladen.\n\n"
            f"{details}"
        )
    if is_missing_model_cache_error(details):
        return (
            "WhisperX konnte die Datei nicht verarbeiten.\n\n"
            "Das gewaehlte Whisper-Modell ist offenbar nicht lokal zwischengespeichert "
            "und konnte nicht heruntergeladen werden. Bitte ein bereits verwendetes Modell "
            "nutzen oder einmalig mit Internetzugang starten.\n\n"
            f"{details}"
        )
    return f"WhisperX konnte die Datei nicht verarbeiten.\n\n{details}"


def is_torchvision_compatibility_error(details: str) -> bool:
    markers = [
        "operator torchvision::nms does not exist",
        "Could not import module 'Wav2Vec2ForCTC'",
    ]
    return all(marker in details for marker in markers)


def is_corrupt_alignment_cache_error(details: str) -> bool:
    if "PytorchStreamReader failed reading zip archive" not in details:
        return False
    if "failed finding central directory" not in details:
        return False
    alignment_markers = [
        "wav2vec2_fairseq_base_ls960_asr_ls960.pth",
        "load_align_model",
        "torchaudio/pipelines/_wav2vec2",
        "load_state_dict_from_url",
    ]
    return any(marker in details for marker in alignment_markers)


def is_missing_model_cache_error(details: str) -> bool:
    markers = [
        "LocalEntryNotFoundError",
        "Failed to resolve 'huggingface.co'",
        "cannot find the appropriate snapshot folder",
        "trying to locate the files on the Hub",
    ]
    return any(marker in details for marker in markers)


def is_silero_download_error(details: str) -> bool:
    markers = [
        "snakers4/silero-vad",
        "torch.hub.load",
        "HTTP Error 403: rate limit exceeded",
        "rate limit exceeded",
    ]
    return any(marker in details for marker in markers)


def load_transcript_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_audio_duration(audio_path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def render_segments(transcript: dict[str, Any], *, merge_adjacent: bool = True) -> list[TranscriptSegment]:
    raw_segments = transcript.get("segments", [])
    rendered: list[TranscriptSegment] = []

    for item in raw_segments:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        rendered.append(
            TranscriptSegment(
                start=_optional_float(item.get("start")),
                end=_optional_float(item.get("end")),
                speaker=str(item.get("speaker") or "SPEAKER_UNKNOWN"),
                text=text,
            )
        )

    if merge_adjacent:
        return merge_adjacent_segments(rendered)
    return rendered


def assign_speakers_by_overlap(
    transcript_segments: list[TranscriptSegment],
    speaker_segments: list[SpeakerSegment],
    *,
    max_merged_duration: float | None = 120.0,
) -> list[TranscriptSegment]:
    assigned = []
    for segment in transcript_segments:
        speaker = best_speaker_for_segment(segment, speaker_segments)
        assigned.append(
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                speaker=speaker or segment.speaker,
                text=segment.text,
            )
        )
    return merge_adjacent_segments(assigned, max_duration=max_merged_duration)


def best_speaker_for_segment(
    transcript_segment: TranscriptSegment,
    speaker_segments: list[SpeakerSegment],
) -> str | None:
    if transcript_segment.start is None or transcript_segment.end is None:
        return None
    overlaps: dict[str, float] = {}
    for speaker_segment in speaker_segments:
        overlap = _overlap_seconds(
            transcript_segment.start,
            transcript_segment.end,
            speaker_segment.start,
            speaker_segment.end,
        )
        if overlap > 0:
            overlaps[speaker_segment.speaker] = overlaps.get(speaker_segment.speaker, 0.0) + overlap
    if not overlaps:
        midpoint = (transcript_segment.start + transcript_segment.end) / 2
        nearest = min(
            speaker_segments,
            key=lambda item: min(abs(item.start - midpoint), abs(item.end - midpoint)),
            default=None,
        )
        return nearest.speaker if nearest else None
    return max(overlaps.items(), key=lambda item: item[1])[0]


def _overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def merge_adjacent_segments(
    segments: list[TranscriptSegment],
    *,
    max_duration: float | None = None,
) -> list[TranscriptSegment]:
    merged: list[TranscriptSegment] = []
    for segment in segments:
        if merged and merged[-1].speaker == segment.speaker and _can_merge(merged[-1], segment, max_duration):
            previous = merged[-1]
            merged[-1] = TranscriptSegment(
                start=previous.start,
                end=segment.end,
                speaker=previous.speaker,
                text=f"{previous.text} {segment.text}".strip(),
            )
        else:
            merged.append(segment)
    return merged


def _can_merge(previous: TranscriptSegment, current: TranscriptSegment, max_duration: float | None) -> bool:
    if max_duration is None:
        return True
    if previous.start is None or current.end is None:
        return True
    return (current.end - previous.start) <= max_duration


def extract_speakers(segments: list[TranscriptSegment]) -> list[str]:
    return sorted({segment.speaker for segment in segments})


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
