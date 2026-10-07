from __future__ import annotations

from ..config import Settings
from .registry import Registry


def build_registry(settings: Settings) -> Registry:
    """Register every provider. Imports here are light: heavy libraries load on first use."""
    from .cloud.anthropic_provider import ClaudeTranslator
    from .cloud.openai_provider import OpenAIASR, OpenAITranslator
    from .cloud.sarvam import SarvamASR, SarvamTranslator, SarvamVoiceClone
    from .local.asr import IndicConformer, WhisperMLX
    from .local.enhance import BasicDenoiser, DeepFilterNetDenoiser, DemucsSeparator, MossFormerDenoiser
    from .local.indictrans import IndicTrans2
    from .local.llm import LocalLLMTranslator
    from .local.voice import ChatterboxTTS, ChatterboxVoiceMatch, MMSMarathiClone

    registry = Registry(settings)
    for provider in (
        # audio clean-up (on this Mac)
        DemucsSeparator(), DeepFilterNetDenoiser(), MossFormerDenoiser(), BasicDenoiser(),
        # speech recognition
        WhisperMLX(), IndicConformer(), SarvamASR(), OpenAIASR(),
        # translation
        IndicTrans2(), LocalLLMTranslator(), ClaudeTranslator(), OpenAITranslator(), SarvamTranslator(),
        # voice-cloned speech and voice matching
        ChatterboxTTS(), MMSMarathiClone(), SarvamVoiceClone(), ChatterboxVoiceMatch(),
    ):
        registry.register(provider)
    return registry
