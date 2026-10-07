from __future__ import annotations

from typing import Callable


def get_pipelines() -> dict[str, Callable]:
    from . import dub, enhance, transcribe

    return {"enhance": enhance.run, "transcribe": transcribe.run, "dub": dub.run}
