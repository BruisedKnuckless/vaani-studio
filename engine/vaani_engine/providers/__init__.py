from __future__ import annotations

from ..config import Settings
from .registry import Registry


def build_registry(settings: Settings) -> Registry:
    """Register every provider. Imports here are light: heavy libraries load on first use."""
    from .local.enhance import BasicDenoiser, DeepFilterNetDenoiser, DemucsSeparator, MossFormerDenoiser

    registry = Registry(settings)
    for provider in (
        DemucsSeparator(),
        DeepFilterNetDenoiser(),
        MossFormerDenoiser(),
        BasicDenoiser(),
    ):
        registry.register(provider)
    return registry
