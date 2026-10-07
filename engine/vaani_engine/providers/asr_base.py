"""Shared logic for speech recognisers that transcribe phrase-sized chunks."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from ..pipelines import vad
from .base import ASR, ProgressFn, Segment, Transcript, noop_progress


class ChunkedASR(ASR):
    """Runs voice-activity detection, then transcribes each phrase on its own.

    Subclasses implement `transcribe_chunk`. For language "auto" they may also
    implement `detect_language`, called once on the first ~30 s of speech."""

    max_chunk_s = 12.0

    @abstractmethod
    def transcribe_chunk(self, audio: np.ndarray, language: str) -> str:
        ...

    def detect_language(self, audio: np.ndarray) -> str:
        return "auto"

    def prepare(self, language: str, progress: ProgressFn) -> None:
        """Load models before the first chunk."""

    def finish(self, segments: list[Segment], language: str, progress: ProgressFn) -> list[Segment]:
        """Post-process all segments (e.g. romanise Hinglish)."""
        return segments

    def transcribe(self, audio_16k: np.ndarray, language: str,
                   progress: ProgressFn = noop_progress) -> Transcript:
        progress(0.01, "Finding speech")
        chunks = vad.speech_chunks(audio_16k, max_len=self.max_chunk_s)
        if not chunks:
            return Transcript(language if language != "auto" else "en", [])
        self.prepare(language, progress)
        if language == "auto":
            first = np.concatenate([vad.cut(audio_16k, s, e) for s, e in chunks[:6]])[: 30 * vad.SR]
            language = self.detect_language(first)
        segments: list[Segment] = []
        for i, (start, end) in enumerate(chunks):
            progress(0.05 + 0.9 * i / len(chunks), f"Transcribing {i + 1}/{len(chunks)}")
            text = self.transcribe_chunk(vad.cut(audio_16k, start, end), language).strip()
            if text:
                segments.append(Segment(round(start, 2), round(end, 2), text))
        segments = self.finish(segments, language, progress)
        progress(1.0, "Transcribed")
        note = self.confidence_note(language)
        return Transcript(language, segments, [note] if note else [])

    def confidence_note(self, language: str) -> str:
        """Override to warn when the recogniser itself was unsure."""
        return ""
