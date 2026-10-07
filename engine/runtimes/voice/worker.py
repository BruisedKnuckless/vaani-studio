"""PyTorch voice runtime: voice conversion (Chatterbox VC) and Marathi speech (Meta MMS-TTS).

Isolated because chatterbox-tts pins torch/transformers versions that differ from the main engine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _protocol import claim_stdout, save_wav, serve  # noqa: E402

claim_stdout()
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

DEVICE = os.environ.get("VAANI_VOICE_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
_state: dict = {}


def _vc(model_dir: str):
    key = ("vc", model_dir)
    if key not in _state:
        from chatterbox.models.s3gen import S3Gen
        from chatterbox.vc import ChatterboxVC

        s3gen = S3Gen()
        s3gen.load_state_dict(torch.load(Path(model_dir) / "s3gen.pt", map_location="cpu", weights_only=True))
        s3gen.to(DEVICE).eval()
        _state[key] = ChatterboxVC(s3gen, DEVICE)
    return _state[key]


def _mms(model_dir: str):
    key = ("mms", model_dir)
    if key not in _state:
        from transformers import AutoTokenizer, VitsModel

        _state[key] = (VitsModel.from_pretrained(model_dir).eval(), AutoTokenizer.from_pretrained(model_dir))
    return _state[key]


def vc(p: dict) -> dict:
    model = _vc(p["model_dir"])
    with torch.inference_mode():
        wav = model.generate(p["source"], target_voice_path=p["ref_audio"])
    return save_wav(p["out_path"], wav, model.sr)


def mms(p: dict) -> dict:
    model, tokenizer = _mms(p["model_dir"])
    inputs = tokenizer(p["text"], return_tensors="pt")
    with torch.inference_mode():
        wav = model(**inputs).waveform[0]
    return save_wav(p["out_path"], wav, model.config.sampling_rate)


def unload(_p: dict) -> dict:
    _state.clear()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return {"ok": True}


if __name__ == "__main__":
    serve({"vc": vc, "mms": mms, "unload": unload})
