from __future__ import annotations

from typing import Callable


def get_pipelines() -> dict[str, Callable]:
    from . import enhance

    return {"enhance": enhance.run}
