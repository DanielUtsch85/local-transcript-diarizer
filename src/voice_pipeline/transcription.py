from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscriptResult:
    text: str
    language: str | None = None
    confidence: float | None = None
    no_speech_prob: float | None = None


class FasterWhisperTranscriber:
    def __init__(self, model: str = "medium", language: str | None = None, device: str = "auto") -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install faster-whisper to enable transcription") from exc
        self.language = language
        self._model = WhisperModel(model, device=device)

    def transcribe(self, path: Path) -> TranscriptResult:
        segments, info = self._model.transcribe(str(path), language=self.language)
        parts: list[str] = []
        avg_logprob: list[float] = []
        no_speech: list[float] = []
        for segment in segments:
            parts.append(segment.text.strip())
            if segment.avg_logprob is not None:
                avg_logprob.append(float(segment.avg_logprob))
            if segment.no_speech_prob is not None:
                no_speech.append(float(segment.no_speech_prob))
        confidence = None
        if avg_logprob:
            confidence = sum(avg_logprob) / len(avg_logprob)
        return TranscriptResult(
            text=" ".join(part for part in parts if part),
            language=getattr(info, "language", self.language),
            confidence=confidence,
            no_speech_prob=sum(no_speech) / len(no_speech) if no_speech else None,
        )
