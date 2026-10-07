"""On-device noise removal and voice/background separation."""

from __future__ import annotations

import gc
import threading

import numpy as np

from ... import audio_io
from ...models_manager import models
from ...pipelines.dsp import match_channels, match_length
from ..base import Denoiser, ProgressFn, Separator, noop_progress
from .torch_utils import shim_torchaudio_backend, torch_device


class BasicDenoiser(Denoiser):
    id = "basic_dsp"
    name = "Basic spectral gate"
    kind = "local"
    license = "MIT"
    rank = 90

    def denoise(self, audio, sr, progress: ProgressFn = noop_progress):
        import noisereduce as nr

        out = []
        for i, channel in enumerate(audio):
            progress(i / len(audio), "Spectral gating")
            out.append(nr.reduce_noise(y=channel, sr=sr, stationary=False, prop_decrease=0.85))
        return np.stack(out).astype(np.float32), sr


class DeepFilterNetDenoiser(Denoiser):
    id = "deepfilternet"
    name = "DeepFilterNet 3"
    kind = "local"
    license = "MIT/Apache-2.0"
    model_ids = ("deepfilternet3",)
    python_modules = ("df",)
    rank = 10

    def __init__(self):
        self._model = None
        self._state = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                shim_torchaudio_backend()
                from df.enhance import init_df

                self._model, self._state, _ = init_df(log_level="ERROR", log_file=None)
        return self._model, self._state

    def denoise(self, audio, sr, progress: ProgressFn = noop_progress):
        import torch

        shim_torchaudio_backend()
        from df.enhance import enhance

        progress(0.05, "Loading DeepFilterNet")
        model, state = self._load()
        model_sr = state.sr()
        x = audio_io.resample(audio, sr, model_sr)
        progress(0.2, "Removing noise")
        # Whole clips at once (no seams); very long files go in 5-minute blocks to cap memory.
        block = 300 * model_sr
        out = []
        for start in range(0, x.shape[-1], block):
            chunk = torch.from_numpy(np.ascontiguousarray(x[:, start:start + block]))
            out.append(enhance(model, state, chunk).numpy())
            progress(0.2 + 0.8 * min(1.0, (start + block) / x.shape[-1]), "Removing noise")
        y = np.concatenate(out, axis=-1).astype(np.float32)
        return match_length(y, x.shape[-1]), model_sr


class MossFormerDenoiser(Denoiser):
    id = "mossformer2"
    name = "MossFormer2 48K (high quality)"
    kind = "local"
    license = "Apache-2.0"
    model_ids = ("mossformer2-se",)
    python_modules = ("mlx_audio",)
    rank = 20
    heavy = True

    def __init__(self):
        self._model = None

    def unload(self):
        self._model = None
        gc.collect()

    def denoise(self, audio, sr, progress: ProgressFn = noop_progress):
        progress(0.02, "Preparing MossFormer2")
        path = models.download("mossformer2-se")
        if self._model is None:
            from mlx_audio.sts.models.mossformer2_se import MossFormer2SEModel

            self._model = MossFormer2SEModel.from_pretrained(str(path))
        x = audio_io.resample(audio, sr, 48000)
        out = []
        for i, channel in enumerate(x):
            progress(0.1 + 0.9 * i / len(x), "Enhancing speech")
            y = np.asarray(self._model.enhance(channel[None, :].astype(np.float32))).reshape(-1)
            out.append(match_length(y[None, :], x.shape[-1])[0])
        return np.stack(out).astype(np.float32), 48000


class DemucsSeparator(Separator):
    id = "demucs"
    name = "Demucs (htdemucs)"
    kind = "local"
    license = "MIT"
    model_ids = ("htdemucs",)
    python_modules = ("demucs",)
    rank = 10
    heavy = True

    def __init__(self):
        self._model = None

    def unload(self):
        self._model = None
        gc.collect()

    def separate(self, audio, sr, progress: ProgressFn = noop_progress):
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        progress(0.02, "Loading Demucs")
        if self._model is None:
            self._model = get_model("htdemucs")
            self._model.eval()
        model = self._model
        x = match_channels(audio_io.resample(audio, sr, model.samplerate), model.audio_channels)
        wav = torch.from_numpy(np.ascontiguousarray(x))
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        wav = (wav - mean) / std
        progress(0.1, "Separating voice from background")
        sources = None
        for device in dict.fromkeys([torch_device(), "cpu"]):
            try:
                with torch.no_grad():
                    sources = apply_model(model, wav[None], device=device, shifts=1, split=True,
                                          overlap=0.25, progress=False)[0]
                break
            except Exception:
                if device == "cpu":
                    raise
        sources = (sources * std + mean).cpu().numpy()
        vocals = sources[model.sources.index("vocals")]
        background = sources.sum(axis=0) - vocals
        progress(1.0, "Separated")
        n_out = audio.shape[-1]
        result = {}
        for name, stem in (("vocals", vocals), ("background", background)):
            stem = audio_io.resample(stem, model.samplerate, sr)
            result[name] = match_length(match_channels(stem, audio.shape[0]), n_out).astype(np.float32)
        return result
