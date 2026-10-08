"""Audio decode/encode with PyAV (FFmpeg libraries bundled in the wheel, so no Homebrew needed),
plus small helpers.

Audio is held as float32 numpy arrays shaped (channels, samples).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

OUTPUT_FORMATS = ("wav", "m4a", "mp3", "flac")
ENCODERS = {"m4a": ("aac", 256_000), "mp3": ("libmp3lame", 320_000)}


@dataclass
class AudioInfo:
    duration: float
    sample_rate: int
    channels: int
    codec: str


def _audio_stream(container, path):
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        raise ValueError(f"No audio in {Path(path).name}")
    return stream


def _channels(stream) -> int:
    layout = getattr(stream.codec_context, "layout", None)
    return int(getattr(layout, "nb_channels", 0) or getattr(stream.codec_context, "channels", 0) or 1)


def probe(path: str | Path) -> AudioInfo:
    import av

    try:
        with av.open(str(path)) as container:
            stream = _audio_stream(container, path)
            if container.duration:
                duration = container.duration / av.time_base
            elif stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            else:
                duration = 0.0
            return AudioInfo(duration=float(duration), sample_rate=int(stream.codec_context.sample_rate or 0),
                             channels=_channels(stream), codec=stream.codec_context.name or "")
    except av.FFmpegError as exc:
        raise RuntimeError(f"Can't read {Path(path).name}: {exc}") from exc


def load(path: str | Path, sr: int = 48000, channels: int | None = None) -> np.ndarray:
    """Decode any audio/video file to float32 (channels, samples) at `sr`.

    channels=None keeps the source layout (capped at stereo).
    """
    import av

    try:
        with av.open(str(path)) as container:
            stream = _audio_stream(container, path)
            if channels is None:
                channels = min(max(_channels(stream), 1), 2)
            resampler = av.AudioResampler(format="fltp", layout="mono" if channels == 1 else "stereo", rate=sr)
            chunks = []
            for frame in container.decode(stream):
                chunks.extend(f.to_ndarray() for f in resampler.resample(frame))
            chunks.extend(f.to_ndarray() for f in resampler.resample(None))
    except av.FFmpegError as exc:
        raise RuntimeError(f"Can't decode {Path(path).name}: {exc}") from exc
    if not chunks:
        return np.zeros((channels, 0), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(chunks, axis=1), dtype=np.float32)


def save(path: str | Path, audio: np.ndarray, sr: int, fmt: str | None = None) -> Path:
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".") or "wav").lower()
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported format {fmt}")
    path = path.with_suffix(f".{fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = as_2d(audio)
    if fmt == "wav":
        sf.write(path, audio.T, sr, subtype="PCM_24")
        return path
    if fmt == "flac":
        sf.write(path, audio.T, sr, subtype="PCM_24", format="FLAC")
        return path

    import av

    codec, bitrate = ENCODERS[fmt]
    layout = "mono" if audio.shape[0] == 1 else "stereo"
    data = np.ascontiguousarray(np.clip(audio[:2], -1.0, 1.0), dtype=np.float32)
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=sr, layout=layout)
        stream.bit_rate = bitrate
        frame = av.AudioFrame.from_ndarray(data, format="fltp", layout=layout)
        frame.sample_rate = sr
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def as_2d(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    return audio[None, :] if audio.ndim == 1 else audio


def to_mono(audio: np.ndarray) -> np.ndarray:
    audio = as_2d(audio)
    return audio.mean(axis=0, keepdims=True)


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return as_2d(audio)
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(sr_from, sr_to)
    return resample_poly(as_2d(audio), sr_to // g, sr_from // g, axis=-1).astype(np.float32)


def peaks(audio: np.ndarray, buckets: int = 1200) -> list[float]:
    """Max-abs envelope for drawing a waveform."""
    mono = np.abs(as_2d(audio)).max(axis=0)
    if mono.size == 0:
        return []
    buckets = max(1, min(buckets, mono.size))
    edges = np.linspace(0, mono.size, buckets + 1, dtype=np.int64)
    return [round(min(1.0, float(mono[a:b].max())) if b > a else 0.0, 4) for a, b in zip(edges[:-1], edges[1:])]


def file_peaks(path: str | Path, buckets: int = 1200) -> dict:
    audio = load(path, sr=8000, channels=1)
    return {"duration": audio.shape[-1] / 8000, "peaks": peaks(audio, buckets)}
