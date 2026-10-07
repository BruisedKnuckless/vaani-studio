"""Audio decode/encode through ffmpeg, plus small helpers.

Audio is held as float32 numpy arrays shaped (channels, samples).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import find_tool

OUTPUT_FORMATS = ("wav", "m4a", "mp3", "flac")


@dataclass
class AudioInfo:
    duration: float
    sample_rate: int
    channels: int
    codec: str


def probe(path: str | Path) -> AudioInfo:
    cmd = [
        find_tool("ffprobe"), "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,codec_name:format=duration",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, check=True, text=True).stdout
    data = json.loads(out)
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(f"No audio stream in {path}")
    s = streams[0]
    return AudioInfo(
        duration=float(data.get("format", {}).get("duration", 0.0)),
        sample_rate=int(s.get("sample_rate", 0)),
        channels=int(s.get("channels", 0)),
        codec=s.get("codec_name", ""),
    )


def load(path: str | Path, sr: int = 48000, channels: int | None = None) -> np.ndarray:
    """Decode any audio/video file to float32 (channels, samples) at `sr`.

    channels=None keeps the source layout (capped at stereo).
    """
    if channels is None:
        channels = min(max(probe(path).channels, 1), 2)
    cmd = [
        find_tool("ffmpeg"), "-nostdin", "-v", "error", "-i", str(path),
        "-vn", "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}: {proc.stderr.decode(errors='ignore')[-500:]}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio.reshape(-1, channels).T.copy()


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
    codec = ["-c:a", "aac", "-b:a", "256k"] if fmt == "m4a" else ["-c:a", "libmp3lame", "-b:a", "320k"]
    cmd = [
        find_tool("ffmpeg"), "-nostdin", "-v", "error", "-y",
        "-f", "f32le", "-ar", str(sr), "-ac", str(audio.shape[0]), "-i", "-",
        *codec, str(path),
    ]
    data = np.ascontiguousarray(audio.T, dtype=np.float32).tobytes()
    proc = subprocess.run(cmd, input=data, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not encode {path}: {proc.stderr.decode(errors='ignore')[-500:]}")
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
