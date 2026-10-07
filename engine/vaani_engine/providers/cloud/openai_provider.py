"""OpenAI: speech recognition (gpt-4o-transcribe) and LLM translation."""

from __future__ import annotations

import json
import os

import numpy as np

from ...config import settings
from .. import translation_prompt as tp
from ..asr_base import ChunkedASR
from ..base import ProgressFn, ProviderError, Translator, noop_progress
from .common import wav_bytes

TRANSCRIBE_MODEL = os.environ.get("VAANI_OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
CHAT_MODEL = os.environ.get("VAANI_OPENAI_MODEL", "gpt-5")


def _client():
    import openai

    return openai.OpenAI(api_key=settings.key("openai"), timeout=300.0, max_retries=3)


def _wrap(exc: Exception) -> ProviderError:
    import openai

    if isinstance(exc, openai.AuthenticationError):
        return ProviderError("OpenAI: the API key was rejected (check it in Settings)")
    if isinstance(exc, openai.RateLimitError):
        return ProviderError("OpenAI: rate limited or out of credits")
    if isinstance(exc, openai.APIConnectionError):
        return ProviderError("OpenAI: can't reach the API (check your internet connection)")
    if isinstance(exc, openai.APIStatusError):
        return ProviderError(f"OpenAI: API error {exc.status_code}")
    return ProviderError(f"OpenAI: {exc}")


class OpenAIASR(ChunkedASR):
    id = "openai_stt"
    name = "OpenAI gpt-4o-transcribe"
    kind = "cloud"
    key_name = "openai"
    languages = frozenset({"auto", "en", "hi", "mr", "hinglish"})
    rank = 20
    max_chunk_s = 25.0

    def transcribe_chunk(self, audio: np.ndarray, language: str) -> str:
        kwargs = {}
        if language in ("en", "hi", "mr"):
            kwargs["language"] = language
        if language == "hinglish":
            kwargs["prompt"] = "Hinglish transcript: write Hindi words in Roman script, e.g. 'aaj office jaana hai'."
        try:
            result = _client().audio.transcriptions.create(
                model=TRANSCRIBE_MODEL, file=("chunk.wav", wav_bytes(audio, 16000), "audio/wav"), **kwargs)
        except Exception as exc:
            raise _wrap(exc) from exc
        return getattr(result, "text", "") or ""


class OpenAITranslator(Translator):
    id = "openai_mt"
    name = "OpenAI GPT"
    kind = "cloud"
    key_name = "openai"
    languages = frozenset({"auto", "en", "hi", "mr", "hinglish"})
    rank = 20
    follows_length = True

    def translate(self, texts, src, tgt, durations=None, progress: ProgressFn = noop_progress):
        client = _client()
        system = tp.system_prompt(src, tgt, dubbing=durations is not None)
        out: list[str] = []
        for offset, batch, batch_durations in tp.batches(texts, durations, size=60):
            progress(offset / max(len(texts), 1), f"Translating {offset + 1}–{offset + len(batch)}")
            try:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": tp.user_message(batch, tgt, batch_durations)}],
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "translations", "schema": tp.SCHEMA, "strict": True}},
                )
            except Exception as exc:
                raise _wrap(exc) from exc
            content = response.choices[0].message.content or ""
            try:
                items = json.loads(content)["translations"]
            except (json.JSONDecodeError, KeyError, TypeError):
                items = tp.parse(content, len(batch))
            if len(items) != len(batch):
                raise ProviderError(f"OpenAI returned {len(items)} lines for {len(batch)}")
            out.extend(str(t) for t in items)
        progress(1.0, "Translated")
        return out
