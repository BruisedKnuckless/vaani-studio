"""Sarvam AI: Indian-language speech recognition, translation and voice-cloned speech.

API reference: https://docs.sarvam.ai/api-reference
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ...config import settings
from ..asr_base import ChunkedASR
from ..base import TTS, ProgressFn, Translator, noop_progress
from .common import request_with_retry, wav_bytes

BASE = "https://api.sarvam.ai"
BCP47 = {"en": "en-IN", "hi": "hi-IN", "mr": "mr-IN", "hinglish": "hi-IN", "auto": "unknown"}
FROM_BCP47 = {"en-IN": "en", "hi-IN": "hi", "mr-IN": "mr"}


def _headers() -> dict:
    return {"api-subscription-key": settings.key("sarvam") or ""}


class SarvamASR(ChunkedASR):
    id = "sarvam_stt"
    name = "Sarvam Saaras"
    kind = "cloud"
    key_name = "sarvam"
    languages = frozenset({"auto", "en", "hi", "mr", "hinglish"})
    rank = 10
    max_chunk_s = 25.0  # the REST endpoint takes up to 30 s per request

    def _request(self, audio: np.ndarray, language: str) -> dict:
        data = {"language_code": BCP47.get(language, "unknown")}
        if language == "hinglish":
            # Romanised output ("mera phone number hai") needs saaras:v3's translit mode.
            data.update(model="saaras:v3", mode="translit")
        else:
            data["model"] = "saaras:v4"
        response = request_with_retry(
            "POST", f"{BASE}/speech-to-text", service="Sarvam", headers=_headers(), data=data,
            files={"file": ("chunk.wav", wav_bytes(audio, 16000), "audio/wav")},
        )
        return response.json()

    def detect_language(self, audio: np.ndarray) -> str:
        code = self._request(audio, "auto").get("language_code")
        return FROM_BCP47.get(code or "", "hi")

    def transcribe_chunk(self, audio: np.ndarray, language: str) -> str:
        return self._request(audio, language).get("transcript", "")


class SarvamTranslator(Translator):
    id = "sarvam_mt"
    name = "Sarvam Mayura"
    kind = "cloud"
    key_name = "sarvam"
    languages = frozenset({"en", "hi", "mr", "hinglish"})
    rank = 30

    def _one(self, text: str, src: str, tgt: str) -> str:
        if not text.strip():
            return ""
        body = {
            "input": text[:1000],
            "source_language_code": "auto" if src in ("auto", "hinglish") else BCP47[src],
            "target_language_code": BCP47[tgt],
            "model": "mayura:v1",
            # Spoken-style output suits subtitles and dubbing better than the default formal register.
            "mode": "code-mixed" if tgt == "hinglish" else "modern-colloquial",
            "numerals_format": "international",
        }
        if tgt == "hinglish":
            body["output_script"] = "roman"
        response = request_with_retry("POST", f"{BASE}/translate", service="Sarvam",
                                      headers=_headers(), json=body)
        return response.json().get("translated_text", "")

    def translate(self, texts, src, tgt, durations=None, progress: ProgressFn = noop_progress):
        done = 0
        out = [""] * len(texts)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(self._one, t, src, tgt): i for i, t in enumerate(texts)}
            for future in futures:
                out[futures[future]] = future.result()
                done += 1
                progress(done / len(texts), f"Translated {done}/{len(texts)}")
        return out


class SarvamVoiceClone(TTS):
    """Bulbul voice cloning: speaks any supported language in the reference speaker's voice.
    Uses a per-request reference clip, so no voice is stored in your Sarvam account."""

    id = "sarvam_tts"
    name = "Sarvam voice clone"
    kind = "cloud"
    key_name = "sarvam"
    languages = frozenset({"en", "hi", "mr"})
    rank = 10

    def synthesize(self, text, language, ref_audio_path, ref_text, max_duration=None):
        import base64
        import io

        import soundfile as sf

        data = {
            "text": text[:1000],
            "language_code": BCP47[language],
            "speech_sample_rate": "24000",
            "output_audio_codec": "wav",
            "enable_text_normalization": "true",
        }
        if ref_text:
            data["ref_text"] = ref_text[:500]
        if max_duration:
            # Sarvam time-stretches the result to fit the dubbing slot.
            data["max_audio_duration"] = f"{max_duration:.2f}"
        with open(ref_audio_path, "rb") as ref:
            response = request_with_retry(
                "POST", f"{BASE}/voices/clone", service="Sarvam", headers=_headers(), data=data,
                files={"ref_audio": ("reference.wav", ref.read(), "audio/wav")}, timeout=180.0,
            )
        audio, sr = sf.read(io.BytesIO(base64.b64decode(response.json()["audio"])), dtype="float32", always_2d=True)
        return audio.T, sr
