"""JSON-lines protocol shared by every runtime worker.

  -> {"id": 1, "method": "tts", "params": {...}}
  <- {"id": 1, "result": {...}}   or   {"id": 1, "error": "..."}

Call `claim_stdout()` before importing heavy libraries so that anything they
print goes to stderr instead of corrupting the protocol channel.
"""

from __future__ import annotations

import json
import sys
import traceback
import warnings
from typing import Callable

_protocol = None


def claim_stdout():
    global _protocol
    if _protocol is None:
        _protocol = sys.stdout
        sys.stdout = sys.stderr
        warnings.filterwarnings("ignore")
    return _protocol


def save_wav(path: str, audio, sr: int) -> dict:
    import numpy as np
    import soundfile as sf

    data = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
    data = np.asarray(data, dtype=np.float32).reshape(-1)
    sf.write(path, data, sr, subtype="PCM_16")
    return {"path": path, "sample_rate": sr, "duration": len(data) / sr}


def serve(methods: dict[str, Callable[[dict], dict]]) -> None:
    out = claim_stdout()
    methods = {"ping": lambda p: {"ok": True}} | methods
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        reply: dict = {"id": request.get("id")}
        try:
            reply["result"] = methods[request["method"]](request.get("params") or {})
        except Exception as exc:  # report every failure to the engine instead of dying
            traceback.print_exc()
            reply["error"] = f"{type(exc).__name__}: {exc}"
        out.write(json.dumps(reply) + "\n")
        out.flush()
