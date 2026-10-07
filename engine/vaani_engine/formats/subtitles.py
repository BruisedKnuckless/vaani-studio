"""SRT, WebVTT and plain-text writers."""

from __future__ import annotations

import textwrap

from ..providers.base import Segment

MAX_LINE = 42


def _stamp(seconds: float, sep: str) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _wrap(text: str) -> str:
    return "\n".join(textwrap.wrap(" ".join(text.split()), MAX_LINE)) or text


def _text(segment: Segment, field: str) -> str:
    return (getattr(segment, field) or "").strip()


def to_srt(segments: list[Segment], field: str = "text") -> str:
    blocks = []
    for i, seg in enumerate((s for s in segments if _text(s, field)), start=1):
        blocks.append(f"{i}\n{_stamp(seg.start, ',')} --> {_stamp(seg.end, ',')}\n{_wrap(_text(seg, field))}\n")
    return "\n".join(blocks)


def to_vtt(segments: list[Segment], field: str = "text") -> str:
    blocks = ["WEBVTT\n"]
    for seg in segments:
        if _text(seg, field):
            blocks.append(f"{_stamp(seg.start, '.')} --> {_stamp(seg.end, '.')}\n{_wrap(_text(seg, field))}\n")
    return "\n".join(blocks)


def to_txt(segments: list[Segment], field: str = "text") -> str:
    return "\n".join(_text(s, field) for s in segments if _text(s, field)) + "\n"
