from __future__ import annotations

import sys
import types


def torch_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def shim_torchaudio_backend() -> None:
    """DeepFilterNet 0.5.x imports `torchaudio.backend.common.AudioMetaData`,
    which newer torchaudio removed. We never use its file I/O, so a stub is enough."""
    if "torchaudio.backend.common" in sys.modules:
        return
    try:
        import torchaudio.backend.common  # noqa: F401
        return
    except ImportError:
        pass
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:  # minimal stand-in, only used in type hints
        def __init__(self, sample_rate: int = 0, num_frames: int = 0, num_channels: int = 0,
                     bits_per_sample: int = 0, encoding: str = ""):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    common.AudioMetaData = AudioMetaData
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common
