"""On-device speech recognition."""

from __future__ import annotations

import gc

import numpy as np

from ...models_manager import models
from ..asr_base import ChunkedASR
from ..base import ProgressFn, Segment

WHISPER_LANGS = {"en": "en", "hi": "hi", "mr": "mr", "hinglish": "hi"}

# Whisper often labels Marathi speech as Hindi. These common words (and the letter ळ)
# separate the two in a transcript even when recognition is rough.
MARATHI_MARKERS = {"आहे", "आहेत", "नाही", "तुम्ही", "तुम्हाला", "आपण", "आपला", "आपले", "आणि", "मध्ये", "म्हणजे",
                   "करतो", "करते", "होते", "होतो", "त्यामुळे", "काय", "कसं", "मला", "आम्ही", "तर", "पण", "तेव्हा",
                   "जर", "नसेल", "असेल", "झाले", "चांगले", "नक्की", "मी"}
HINDI_MARKERS = {"है", "हैं", "था", "थे", "नहीं", "और", "में", "का", "की", "के", "को", "से", "आप", "हम", "यह",
                 "वह", "क्या", "कि", "भी", "तो", "मैं", "हूँ", "रहा", "रही"}


def marathi_score(text: str) -> float:
    """Above 0 means the Devanagari text reads more like Marathi than Hindi."""
    words = [w.strip(".,!?।\"'") for w in text.split()]
    mr = sum(w in MARATHI_MARKERS or w.startswith("तुम्ह") for w in words) + 2 * text.count("ळ")
    hi = sum(w in HINDI_MARKERS for w in words)
    return (mr - hi) / max(len(words), 1)


class WhisperMLX(ChunkedASR):
    """Whisper large-v3 turbo on Apple's MLX. Hinglish = Hindi transcription
    romanised by the on-device LLM."""

    id = "whisper_mlx"
    name = "Whisper large-v3 turbo"
    kind = "local"
    license = "MIT"
    languages = frozenset({"auto", "en", "hi", "mr", "hinglish"})
    model_ids = ("whisper-large-v3-turbo",)
    python_modules = ("mlx_whisper",)
    heavy = True

    def rank_for(self, src, tgt):
        return {"mr": 40, "hinglish": 30}.get(src or "", 10)

    def models_for(self, src=None, tgt=None):
        return self.model_ids + (("local-llm",) if src == "hinglish" else ())

    def advice(self, src=None, tgt=None):
        if src == "mr":
            return "Whisper turbo often mishears Marathi; pick Whisper large-v2 Marathi or Sarvam instead"
        return ""

    def prepare(self, language: str, progress: ProgressFn) -> None:
        progress(0.03, "Loading Whisper")
        self._path = str(models.download("whisper-large-v3-turbo"))
        self._logprobs: list[float] = []

    def _run(self, audio: np.ndarray, language: str | None) -> dict:
        import mlx_whisper

        if language == "mr" and models.is_downloaded("whisper-marathi"):
            # Auto-detected Marathi: use the Marathi fine-tune when it's on disk.
            return marathi_decode(audio, str(models.path("whisper-marathi")))
        return mlx_whisper.transcribe(
            audio.astype(np.float32), path_or_hf_repo=self._path, language=language,
            condition_on_previous_text=False, verbose=None, temperature=(0.0, 0.2, 0.4),
        )

    def detect_language(self, audio: np.ndarray) -> str:
        result = self._run(audio, None)
        code = {"ur": "hi"}.get(result.get("language", "en"), result.get("language", "en"))  # Hindi is sometimes tagged Urdu
        if code in ("hi", "mr"):
            code = "mr" if marathi_score(result.get("text", "")) > 0.02 else "hi"
        return code

    def transcribe_chunk(self, audio: np.ndarray, language: str) -> str:
        result = self._run(audio, WHISPER_LANGS.get(language, language))
        kept = [s for s in result.get("segments", [])
                if not (s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0)]
        self._logprobs.extend(s.get("avg_logprob", 0.0) for s in kept)
        return " ".join(s["text"].strip() for s in kept)

    def confidence_note(self, language: str) -> str:
        """Whisper scores clear speech around -0.1 to -0.3; much lower means it was guessing."""
        if not self._logprobs:
            return ""
        mean = sum(self._logprobs) / len(self._logprobs)
        if mean < -0.5:
            return (f"Speech recognition confidence was low ({mean:.2f}); the transcript may be inaccurate"
                    + (". Whisper is weak at Marathi: Sarvam or IndicConformer will do much better" if language == "mr" else ""))
        return ""

    def finish(self, segments: list[Segment], language: str, progress: ProgressFn) -> list[Segment]:
        if language != "hinglish" or not segments:
            return segments
        from .llm import LocalLLMTranslator

        progress(0.96, "Writing Hinglish in Roman script")
        roman = LocalLLMTranslator().translate([s.text for s in segments], "hi", "hinglish")
        for seg, text in zip(segments, roman):
            seg.text = text or seg.text
        return segments

    def unload(self):
        try:
            from mlx_whisper.transcribe import ModelHolder

            ModelHolder.model = None
            ModelHolder.model_path = None
        except Exception:
            pass
        gc.collect()


def marathi_decode(audio: np.ndarray, path: str) -> dict:
    """Decode settings for the Marathi fine-tune: one temperature fallback, and a token cap so a
    repetition loop can't run to the model's limit (loops are trimmed afterwards)."""
    import mlx_whisper

    from ..translation_prompt import strip_loops

    seconds = len(audio) / 16000
    result = mlx_whisper.transcribe(
        audio.astype(np.float32), path_or_hf_repo=path, language="mr", condition_on_previous_text=False,
        verbose=None, temperature=(0.0, 0.3), sample_len=min(224, int(seconds * 30) + 30),
    )
    for seg in result.get("segments", []):
        seg["text"] = strip_loops(seg["text"])
    return result


class WhisperMarathi(WhisperMLX):
    """Whisper large-v2 fine-tuned on Marathi, converted to MLX. Far more accurate on Marathi
    than the stock model."""

    id = "whisper_marathi"
    name = "Whisper large-v2 Marathi"
    languages = frozenset({"mr"})
    model_ids = ("whisper-marathi",)
    license = "Apache-2.0"

    def rank_for(self, src, tgt):
        return 10

    def models_for(self, src=None, tgt=None):
        return self.model_ids

    def advice(self, src=None, tgt=None):
        return ""

    def prepare(self, language: str, progress: ProgressFn) -> None:
        progress(0.03, "Loading Whisper Marathi")
        self._path = str(models.download("whisper-marathi"))
        self._logprobs = []

    def _run(self, audio: np.ndarray, language: str | None) -> dict:
        return marathi_decode(audio, self._path)


class IndicConformer(ChunkedASR):
    """AI4Bharat IndicConformer 600M (22 Indian languages). Best local option for Marathi."""

    id = "indic_conformer"
    name = "IndicConformer 600M"
    kind = "local"
    license = "MIT"
    languages = frozenset({"mr", "hi"})
    model_ids = ("indic-conformer",)
    python_modules = ("onnxruntime", "transformers")
    heavy = True

    def __init__(self):
        self._model = None

    def rank_for(self, src, tgt):
        return 15 if src == "mr" else 20

    def prepare(self, language: str, progress: ProgressFn) -> None:
        if self._model is None:
            progress(0.03, "Loading IndicConformer")
            from transformers import AutoModel

            path = models.download("indic-conformer")
            self._model = AutoModel.from_pretrained(str(path), trust_remote_code=True)

    def transcribe_chunk(self, audio: np.ndarray, language: str) -> str:
        import torch

        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None, :]
        with torch.no_grad():
            text = self._model(wav, language, "rnnt")
        return text if isinstance(text, str) else str(text[0] if text else "")

    def unload(self):
        self._model = None
        gc.collect()
