"""On-device voice cloning, run in isolated runtimes:
  mlxvoice  Chatterbox multilingual on Apple's GPU (English, Hindi)
  voice     Chatterbox voice conversion + Meta MMS-TTS Marathi (PyTorch)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ... import audio_io
from ...models_manager import models
from ...runtimes.worker_host import WorkerHost
from ..base import TTS, VoiceConverter


class _VoiceRuntime:
    runtime = "voice"

    def readiness(self, src=None, tgt=None):
        if not WorkerHost.get(self.runtime).installed:
            return "unavailable", "The voice runtime isn't installed (run scripts/bootstrap.sh)"
        return super().readiness(src, tgt)

    def unload(self):
        WorkerHost.get(self.runtime).stop()

    @staticmethod
    def _tmp(suffix: str = ".wav") -> str:
        return tempfile.NamedTemporaryFile(prefix="vaani-", suffix=suffix, delete=False).name

    @staticmethod
    def _read(path: str) -> tuple[np.ndarray, int]:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        Path(path).unlink(missing_ok=True)
        return audio.T, sr


class ChatterboxTTS(_VoiceRuntime, TTS):
    runtime = "mlxvoice"
    id = "chatterbox"
    name = "Chatterbox voice clone"
    kind = "local"
    license = "MIT"
    languages = frozenset({"en", "hi"})
    model_ids = ("chatterbox-mlx",)
    rank = 10
    heavy = True

    def synthesize(self, text, language, ref_audio_path, ref_text, max_duration=None):
        out = self._tmp()
        WorkerHost.get(self.runtime).call("tts", {
            "model_dir": str(models.download("chatterbox-mlx")), "text": text, "language": language,
            "ref_audio": ref_audio_path, "out_path": out, "exaggeration": 0.5, "cfg_weight": 0.5,
            "temperature": 0.7,
        })
        return self._read(out)


class MMSMarathiClone(_VoiceRuntime, TTS):
    """Meta MMS-TTS speaks the Marathi text; Chatterbox VC then gives it the speaker's voice."""

    id = "mms_vc"
    name = "MMS Marathi + voice match"
    kind = "local"
    license = "CC-BY-NC-4.0 + MIT"
    languages = frozenset({"mr"})
    model_ids = ("mms-tts-mar", "chatterbox-vc")
    rank = 30
    heavy = True

    def synthesize(self, text, language, ref_audio_path, ref_text, max_duration=None):
        host = WorkerHost.get(self.runtime)
        plain = self._tmp()
        host.call("mms", {"model_dir": str(models.download("mms-tts-mar")), "text": text, "out_path": plain})
        out = self._tmp()
        host.call("vc", {"model_dir": str(models.download("chatterbox-vc")), "source": plain,
                         "ref_audio": ref_audio_path, "out_path": out})
        Path(plain).unlink(missing_ok=True)
        return self._read(out)


class ChatterboxVoiceMatch(_VoiceRuntime, VoiceConverter):
    id = "chatterbox_vc"
    name = "Chatterbox voice match"
    kind = "local"
    license = "MIT"
    model_ids = ("chatterbox-vc",)
    heavy = True

    def convert(self, audio, sr, ref_audio_path):
        source = self._tmp()
        audio_io.save(source, audio_io.to_mono(audio), sr, "wav")
        out = self._tmp()
        WorkerHost.get(self.runtime).call("vc", {"model_dir": str(models.download("chatterbox-vc")),
                                                 "source": source, "ref_audio": ref_audio_path, "out_path": out})
        Path(source).unlink(missing_ok=True)
        return self._read(out)
