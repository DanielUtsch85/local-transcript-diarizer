from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptSegment:
    start: float | None
    end: float | None
    speaker: str
    text: str

    @property
    def timestamp(self) -> str:
        if self.start is None:
            return "??:??"
        return format_timestamp(self.start)


@dataclass(frozen=True)
class SpeakerSegment:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class SpeakerMapping:
    labels: dict[str, str]

    def label_for(self, speaker: str) -> str:
        label = self.labels.get(speaker, speaker).strip()
        return label or speaker


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
