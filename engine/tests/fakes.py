"""Fake providers for testing routing without models or network."""

from __future__ import annotations

import numpy as np

from vaani_engine.config import Settings
from vaani_engine.providers.base import ASR, TTS, Denoiser, Segment, Separator, Transcript, Translator
from vaani_engine.providers.registry import Registry

INDIC = frozenset({"auto", "en", "hi", "mr", "hinglish"})


class FakeDenoiser(Denoiser):
    id, name, kind = "fake_denoise", "Fake denoiser", "local"

    def denoise(self, audio, sr, progress=None):
        return audio * 0.5, sr


class FakeSeparator(Separator):
    id, name, kind = "fake_sep", "Fake separator", "local"

    def separate(self, audio, sr, progress=None):
        return {"vocals": audio, "background": np.zeros_like(audio)}


class FakeLocalASR(ASR):
    id, name, kind = "local_asr", "Local ASR", "local"
    languages = frozenset({"auto", "en", "hi", "mr", "hinglish"})

    def transcribe(self, audio_16k, language, progress=None):
        return Transcript(language, [Segment(0.0, 1.0, "namaste")])


class FakeCloudASR(ASR):
    id, name, kind, key_name = "cloud_asr", "Cloud ASR", "cloud", "sarvam"
    languages = INDIC

    def transcribe(self, audio_16k, language, progress=None):
        return Transcript(language, [Segment(0.0, 1.0, "namaste (cloud)")])


class FailingCloudASR(FakeCloudASR):
    def transcribe(self, audio_16k, language, progress=None):
        raise RuntimeError("HTTP 500")


class FakeLocalMT(Translator):
    id, name, kind = "local_mt", "Local MT", "local"
    languages = frozenset({"en", "hi", "mr"})

    def translate(self, texts, src, tgt, durations=None, progress=None):
        return [f"[{tgt}] {t}" for t in texts]


class FakeLLM(Translator):
    id, name, kind, key_name = "llm_mt", "Cloud LLM", "cloud", "anthropic"
    languages = INDIC
    rank = 10

    def translate(self, texts, src, tgt, durations=None, progress=None):
        return [f"<{tgt}> {t}" for t in texts]


class FakeLocalTTS(TTS):
    id, name, kind = "local_tts", "Local TTS", "local"
    languages = frozenset({"en", "hi", "mr"})

    def synthesize(self, text, language, ref_audio_path, ref_text, speed=1.0):
        return np.zeros((1, 24000), dtype=np.float32), 24000


class FakeCloudTTS(TTS):
    id, name, kind, key_name = "cloud_tts", "Cloud TTS", "cloud", "sarvam"
    languages = frozenset({"en", "hi", "mr"})

    def synthesize(self, text, language, ref_audio_path, ref_text, speed=1.0):
        return np.zeros((1, 24000), dtype=np.float32), 24000


def make_registry(keys: dict | None = None, offline: bool = False, failing_cloud_asr: bool = False) -> Registry:
    registry = Registry(Settings(keys=dict(keys or {}), offline=offline))
    cloud_asr = FailingCloudASR() if failing_cloud_asr else FakeCloudASR()
    for p in (FakeDenoiser(), FakeSeparator(), FakeLocalASR(), cloud_asr, FakeLocalMT(), FakeLLM(),
              FakeLocalTTS(), FakeCloudTTS()):
        registry.register(p)
    return registry
