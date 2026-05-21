from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

from .models import SpeakerSegment


def run_local_diarize(
    audio_path: Path,
    min_speakers: int | None,
    max_speakers: int | None,
) -> list[SpeakerSegment]:
    try:
        from diarize import diarize
    except ImportError as exc:
        raise RuntimeError(
            "Das lokale Sprecher-Backend `diarize` ist nicht installiert. "
            "Bitte `python -m pip install diarize` ausfuehren."
        ) from exc

    minimum = min_speakers or 1
    maximum = max_speakers or max(20, minimum)
    exact = minimum if min_speakers and max_speakers and min_speakers == max_speakers else None

    try:
        with tempfile.TemporaryDirectory(prefix="local-diarize-") as temp_dir:
            diarize_audio_path = prepare_audio_for_local_diarize(audio_path, Path(temp_dir))
            result = diarize(
                diarize_audio_path,
                min_speakers=minimum,
                max_speakers=maximum,
                num_speakers=exact,
            )
    except subprocess.CalledProcessError as exc:
        details = exc.stderr or exc.stdout or str(exc)
        raise RuntimeError(format_local_diarize_error(details)) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            format_local_diarize_error("ffmpeg wurde nicht gefunden. Die lokale Sprechererkennung kann m4a/mp4 nicht vorbereiten.")
        ) from exc
    except RuntimeError as exc:
        raise RuntimeError(format_local_diarize_error(str(exc))) from exc

    return [
        SpeakerSegment(start=float(segment.start), end=float(segment.end), speaker=str(segment.speaker))
        for segment in result.segments
    ]


def prepare_audio_for_local_diarize(audio_path: Path, temp_dir: Path) -> Path:
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    converted = temp_dir / f"{audio_path.stem}-diarize.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(converted),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return converted


def format_local_diarize_error(details: str) -> str:
    if is_corrupt_silero_vad_error(details):
        return (
            "Die Transkription wurde erstellt, aber die lokale Sprechererkennung ist beim Laden von Silero VAD "
            "gescheitert.\n\n"
            "Die installierte Silero-Modelldatei scheint beschaedigt oder unlesbar zu sein. Bitte `silero-vad` "
            "in der virtuellen Umgebung neu installieren, oder den Lauf vorerst mit `Nur transkribieren` starten.\n\n"
            f"{details}"
        )
    return f"Lokale Sprechererkennung fehlgeschlagen.\n\n{details}"


def is_corrupt_silero_vad_error(details: str) -> bool:
    markers = [
        "PytorchStreamReader failed reading zip archive",
        "failed finding central directory",
    ]
    return all(marker in details for marker in markers)
