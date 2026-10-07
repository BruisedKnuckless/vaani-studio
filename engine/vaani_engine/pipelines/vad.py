"""Speech detection: split audio into phrase-sized chunks.

Every speech recogniser works on these chunks, so subtitles and dubbing get the
same segment boundaries whichever provider (local or cloud) transcribes them.
"""

from __future__ import annotations

import threading

import numpy as np

SR = 16000
_model = None
_lock = threading.Lock()


def _silero():
    global _model
    with _lock:
        if _model is None:
            from silero_vad import load_silero_vad

            _model = load_silero_vad()
    return _model


def speech_regions(audio: np.ndarray, min_silence_ms: int = 280) -> list[tuple[float, float]]:
    """Raw speech regions in seconds."""
    import torch
    from silero_vad import get_speech_timestamps

    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    stamps = get_speech_timestamps(torch.from_numpy(mono), _silero(), sampling_rate=SR,
                                   min_silence_duration_ms=min_silence_ms, speech_pad_ms=80,
                                   return_seconds=True)
    return [(float(s["start"]), float(s["end"])) for s in stamps]


def group_regions(regions: list[tuple[float, float]], max_len: float = 12.0, max_gap: float = 0.6,
                  min_len: float = 1.2) -> list[tuple[float, float]]:
    """Merge nearby regions into phrases no longer than `max_len`; split anything longer."""
    chunks: list[list[float]] = []
    for start, end in regions:
        if chunks:
            last = chunks[-1]
            gap = start - last[1]
            short = (last[1] - last[0]) < min_len
            if (gap <= max_gap or short) and end - last[0] <= max_len:
                last[1] = end
                continue
        chunks.append([start, end])
    out: list[tuple[float, float]] = []
    for start, end in chunks:
        length = end - start
        if length <= max_len:
            out.append((start, end))
            continue
        parts = int(np.ceil(length / max_len))
        step = length / parts
        out.extend((start + i * step, start + (i + 1) * step) for i in range(parts))
    return out


def speech_chunks(audio: np.ndarray, max_len: float = 12.0) -> list[tuple[float, float]]:
    regions = speech_regions(audio)
    if not regions:
        return []
    return group_regions(regions, max_len=max_len)


def cut(audio: np.ndarray, start: float, end: float, pad: float = 0.1) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    a = max(0, int((start - pad) * SR))
    b = min(mono.size, int((end + pad) * SR))
    return mono[a:b]
