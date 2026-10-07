"""Provider interfaces. Every stage of every pipeline is served by a provider
that runs either on this Mac ("local") or through an API ("cloud")."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Literal

import numpy as np

ProgressFn = Callable[[float, str], None]

LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "hinglish": "Hinglish",
}


def noop_progress(_fraction: float, _message: str = "") -> None:
    pass


class Stage(str, Enum):
    SEPARATE = "separate"
    DENOISE = "denoise"
    POLISH = "polish"
    ASR = "asr"
    TRANSLATE = "translate"
    TTS = "tts"
    VOICE_MATCH = "voice_match"


STAGE_LABELS = {
    Stage.SEPARATE: "Separate voice",
    Stage.DENOISE: "Remove noise",
    Stage.POLISH: "Polish & loudness",
    Stage.ASR: "Transcribe",
    Stage.TRANSLATE: "Translate",
    Stage.TTS: "Voice (TTS)",
    Stage.VOICE_MATCH: "Voice match",
}

Kind = Literal["local", "cloud"]


class ProviderError(RuntimeError):
    """A provider failed in a way the user should hear about."""


@dataclass
class Segment:
    start: float
    end: float
    text: str
    translation: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Transcript:
    language: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def to_dict(self) -> dict:
        return {"language": self.language, "segments": [s.to_dict() for s in self.segments]}

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        return cls(language=data.get("language", "auto"),
                   segments=[Segment(**s) for s in data.get("segments", [])])


class Provider(ABC):
    id: str = ""
    name: str = ""
    stage: Stage
    kind: Kind = "local"
    key_name: str | None = None
    # Languages this provider handles. For ASR: source; TTS: target;
    # translate: both ends. None means any language (e.g. audio cleanup).
    languages: frozenset[str] | None = None
    license: str = ""
    # Model catalog ids this provider needs on disk (local providers).
    model_ids: tuple[str, ...] = ()
    # Lower runs first when several providers of the same kind qualify.
    rank: int = 50
    # Heavy providers are unloaded after each job to keep memory free.
    heavy: bool = False

    def unload(self) -> None:
        """Release models from memory."""

    def supports(self, src: str | None, tgt: str | None) -> bool:
        if self.languages is None:
            return True
        if self.stage == Stage.ASR:
            return src in self.languages
        if self.stage == Stage.TTS:
            return tgt in self.languages
        if self.stage == Stage.TRANSLATE:
            # The source isn't known until speech is recognised, so "auto" only checks the target.
            return (src == "auto" or src in self.languages) and tgt in self.languages
        return True

    def rank_for(self, src: str | None, tgt: str | None) -> int:
        """Ordering among providers of the same kind; override for per-language strengths."""
        return self.rank

    # Python packages that must be importable (checked without importing them).
    python_modules: tuple[str, ...] = ()

    def models_for(self, src: str | None = None, tgt: str | None = None) -> tuple[str, ...]:
        """Model catalog ids needed for this language pair (defaults to all of model_ids)."""
        return self.model_ids

    def readiness(self, src: str | None = None, tgt: str | None = None) -> tuple[str, str]:
        """("ready" | "needs_download" | "unavailable", human note)."""
        from importlib.util import find_spec

        from ..models_manager import readiness_for

        missing = [m for m in self.python_modules if not find_spec(m)]
        if missing:
            return "unavailable", f"{', '.join(missing)} isn't installed (run scripts/bootstrap.sh)"
        return readiness_for(self.models_for(src, tgt))

    def describe(self) -> dict:
        state, note = self.readiness()
        return {
            "id": self.id, "name": self.name, "stage": self.stage.value, "kind": self.kind,
            "key_name": self.key_name, "languages": sorted(self.languages) if self.languages else None,
            "license": self.license, "models": list(self.model_ids), "state": state, "note": note,
        }


class Separator(Provider):
    stage = Stage.SEPARATE

    @abstractmethod
    def separate(self, audio: np.ndarray, sr: int, progress: ProgressFn = noop_progress) -> dict[str, np.ndarray]:
        """Return {"vocals": ..., "background": ...} at the input rate and channel count."""


class Denoiser(Provider):
    stage = Stage.DENOISE

    @abstractmethod
    def denoise(self, audio: np.ndarray, sr: int, progress: ProgressFn = noop_progress) -> tuple[np.ndarray, int]:
        """Return (clean audio, sample rate). Must keep timing aligned with the input."""


class ASR(Provider):
    stage = Stage.ASR

    @abstractmethod
    def transcribe(self, audio_16k: np.ndarray, language: str,
                   progress: ProgressFn = noop_progress) -> Transcript:
        """audio_16k is mono float32 at 16 kHz."""


class Translator(Provider):
    stage = Stage.TRANSLATE
    # True when the translator honours `durations` (LLMs), so dubbing can ask for shorter lines.
    follows_length: bool = False

    @abstractmethod
    def translate(self, texts: list[str], src: str, tgt: str,
                  durations: list[float] | None = None,
                  progress: ProgressFn = noop_progress) -> list[str]:
        """Translate line by line. `durations` (seconds per line) asks for
        dubbing-friendly output that can be spoken within that time."""


class TTS(Provider):
    stage = Stage.TTS

    @abstractmethod
    def synthesize(self, text: str, language: str, ref_audio_path: str, ref_text: str,
                   max_duration: float | None = None) -> tuple[np.ndarray, int]:
        """Speak `text` in the voice of the reference clip. Providers that can fit
        speech to a length use `max_duration`; the dub pipeline time-fits the rest."""

    def prepare(self, ref_audio_path: str, ref_text: str, language: str) -> None:
        """Optional one-time setup per job (e.g. registering a cloned voice)."""


class VoiceConverter(Provider):
    stage = Stage.VOICE_MATCH

    @abstractmethod
    def convert(self, audio: np.ndarray, sr: int, ref_audio_path: str) -> tuple[np.ndarray, int]:
        ...
