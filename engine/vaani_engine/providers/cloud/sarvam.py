"""Sarvam AI: Indian-language speech recognition, translation and voice-cloned speech.

API reference: https://docs.sarvam.ai/api-reference
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ...config import settings
from ..asr_base import ChunkedASR
from .. import translation_prompt as tp
from ..base import TTS, ProgressFn, ProviderError, Translator, noop_progress
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
    max_chunk_s = 15.0  # the endpoint takes up to 30 s; shorter lines make better subtitles and dub timing

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


class SarvamLLMTranslator(Translator):
    """Sarvam-105B (chat model, strong in Indian languages) with Vaani's translation instructions:
    fluent, idiomatic output that translates meaning rather than words, and fits dubbing time slots."""

    id = "sarvam_llm"
    name = "Sarvam-105B"
    kind = "cloud"
    key_name = "sarvam"
    languages = frozenset({"auto", "en", "hi", "mr", "hinglish"})
    rank = 25
    follows_length = True

    # Reasoning tokens count against max_tokens, so translation asks for little reasoning and, if the
    # model still runs out of room, retries without reasoning and then with smaller batches.
    EFFORTS = ("low", None)
    MAX_TOKENS = 16384

    def translate(self, texts, src, tgt, durations=None, progress: ProgressFn = noop_progress):
        system = tp.system_prompt(src, tgt, dubbing=durations is not None)
        out: list[str] = []
        for offset, batch, batch_durations in tp.batches(texts, durations, size=30):
            progress(offset / max(len(texts), 1), f"Translating {offset + 1}–{offset + len(batch)}")
            out.extend(self._translate_batch(system, batch, tgt, batch_durations))
        progress(1.0, "Translated")
        return out

    def _translate_batch(self, system, batch, tgt, durations) -> list[str]:
        for effort in self.EFFORTS:
            items = self._request(system, batch, tgt, durations, effort)
            if items is not None:
                return items
        if len(batch) > 1:
            half = len(batch) // 2
            return (self._translate_batch(system, batch[:half], tgt, durations[:half] if durations else None)
                    + self._translate_batch(system, batch[half:], tgt, durations[half:] if durations else None))
        raise ProviderError("Sarvam-105B kept running out of output space on this line; try Translate with "
                            "another service (Advanced) or shorten the line")

    def _request(self, system, batch, tgt, durations, effort) -> list[str] | None:
        """One chat request. Returns None when the model ran out of tokens before answering."""
        import json
        import logging

        key = settings.key("sarvam") or ""
        headers = {"api-subscription-key": key, "Authorization": f"Bearer {key}"}
        body = {
            "model": "sarvam-105b",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": tp.user_message(batch, tgt, durations)}],
            "temperature": 0.3,
            "max_tokens": self.MAX_TOKENS,
            "reasoning_effort": effort,  # None is sent as null, which turns reasoning off
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "translations", "schema": tp.SCHEMA, "strict": True}},
        }
        data = request_with_retry("POST", f"{BASE}/v1/chat/completions", service="Sarvam",
                                  headers=headers, json=body, timeout=300.0).json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if choice.get("finish_reason") == "length":
            logging.getLogger("vaani").warning(
                "Sarvam-105B ran out of tokens (effort=%s, completion=%s, reasoning=%s); retrying",
                effort, usage.get("completion_tokens"), reasoning)
            return None
        content = (choice.get("message") or {}).get("content") or ""
        try:
            items = json.loads(content)["translations"]
        except (json.JSONDecodeError, KeyError, TypeError):
            items = tp.parse(content, len(batch))
        if len(items) != len(batch):
            raise ProviderError(f"Sarvam-105B returned {len(items)} lines for {len(batch)}")
        return [str(t) for t in items]


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

    def synthesize(self, text, language, ref_audio_path, ref_text, max_duration=None, voice=None, pace=1.0):
        import base64
        import io

        import soundfile as sf

        data = {
            "text": text[:1000],
            "language_code": BCP47[language],
            "speech_sample_rate": "48000",  # full-band output; the model renders at this rate
            "output_audio_codec": "wav",
            "enable_text_normalization": "true",
        }
        if ref_text:
            data["ref_text"] = ref_text[:500]
        if max_duration:
            # Sarvam time-stretches the result to fit the dubbing slot (can't be combined with pace).
            data["max_audio_duration"] = f"{max_duration:.2f}"
        elif abs(pace - 1.0) > 0.01:
            data["pace"] = f"{min(max(pace, 0.5), 2.0):.2f}"
        with open(ref_audio_path, "rb") as ref:
            response = request_with_retry(
                "POST", f"{BASE}/voices/clone", service="Sarvam", headers=_headers(), data=data,
                files={"ref_audio": ("reference.wav", ref.read(), "audio/wav")}, timeout=180.0,
            )
        audio, sr = sf.read(io.BytesIO(base64.b64decode(response.json()["audio"])), dtype="float32", always_2d=True)
        return audio.T, sr


STUDIO_VOICES = ("shubh", "aditya", "ritu", "priya", "neha", "rahul", "pooja", "rohan", "simran", "kavya", "amit",
                 "dev", "ishita", "shreya", "ratan", "varun", "manan", "sumit", "roopa", "kabir", "aayan",
                 "ashutosh", "advait", "anand", "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti",
                 "suhani", "mohit", "kavitha", "rehan", "soham", "rupali")


STUDIO_PACE = 0.92

# Sarvam marks these two as its top-rated, all-round voices.
RECOMMENDED_VOICES = ("shubh", "ishita")

# Gender-neutral sample lines for voice previews.
PREVIEW_TEXT = {
    "en": "Hello! This is a sample of my voice. Your translated audio will be read in exactly this voice.",
    "hi": "नमस्ते! यह मेरी आवाज़ का एक नमूना है। आपका अनुवादित ऑडियो इसी आवाज़ में सुनाई देगा।",
    "mr": "नमस्कार! हा माझ्या आवाजाचा एक नमुना आहे. तुमचा अनुवादित ऑडिओ याच आवाजात ऐकू येईल.",
}


class SarvamStudioVoice(TTS):
    """Bulbul v3 preset voices: very natural, but not the original speaker's voice."""

    id = "sarvam_bulbul"
    name = "Sarvam studio voice"
    kind = "cloud"
    key_name = "sarvam"
    languages = frozenset({"en", "hi", "mr"})
    rank = 50  # only when chosen: dubbing normally keeps the speaker's own voice

    def synthesize(self, text, language, ref_audio_path, ref_text, max_duration=None, voice=None, pace=1.0):
        import base64
        import io

        import soundfile as sf

        body = {
            "text": text[:2500],
            "language_code": BCP47[language],
            "model": "bulbul:v3",
            "speaker": voice if voice in STUDIO_VOICES else "shubh",
            # Studio voices default to a brisk ~165 words/min; 0.92× gives a calmer ~150 for narration.
            "pace": round(min(max(STUDIO_PACE * pace, 0.5), 2.0), 2),
            "speech_sample_rate": 48000,
            "output_audio_codec": "wav",
        }
        response = request_with_retry("POST", f"{BASE}/text-to-speech", service="Sarvam",
                                      headers=_headers(), json=body, timeout=120.0)
        audio, sr = sf.read(io.BytesIO(base64.b64decode(response.json()["audios"][0])), dtype="float32",
                            always_2d=True)
        return audio.T, sr
