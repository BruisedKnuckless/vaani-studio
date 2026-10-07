"""MLX voice runtime: voice-cloned speech with Chatterbox multilingual on Apple's GPU.

Isolated because mlx-audio 0.5 needs a newer transformers than the main engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _protocol import claim_stdout, save_wav, serve  # noqa: E402

claim_stdout()

import numpy as np  # noqa: E402

_state: dict = {}


def _model(model_dir: str):
    if _state.get("dir") != model_dir:
        from mlx_audio.tts.utils import load_model

        _state.clear()
        _state["model"] = load_model(model_dir)
        _state["dir"] = model_dir
    return _state["model"]


def tts(p: dict) -> dict:
    model = _model(p["model_dir"])
    chunks = [np.array(r.audio) for r in model.generate(
        p["text"], ref_audio=p["ref_audio"], lang_code=p["language"], verbose=False,
        exaggeration=p.get("exaggeration", 0.5), cfg_weight=p.get("cfg_weight", 0.5),
        temperature=p.get("temperature", 0.7), speed=p.get("speed", 1.0),
    )]
    if not chunks:
        raise ValueError("No audio was generated")
    return save_wav(p["out_path"], np.concatenate(chunks), model.sample_rate)


def unload(_p: dict) -> dict:
    _state.clear()
    return {"ok": True}


if __name__ == "__main__":
    serve({"tts": tts, "unload": unload})
