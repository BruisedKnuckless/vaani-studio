"""Built-in speech polish: EQ, de-esser, compression and loudness normalisation."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

LOUDNESS_PRESETS = {
    "podcast": -16.0,
    "social": -14.0,
    "broadcast": -23.0,
}
TRUE_PEAK_DB = -1.0


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20))


def loudness(audio: np.ndarray, sr: int) -> float | None:
    """Integrated loudness in LUFS, or None for silence / very short clips."""
    import pyloudnorm as pyln

    if audio.shape[-1] < int(0.5 * sr):
        return None
    value = pyln.Meter(sr).integrated_loudness(audio.T.astype(np.float64))
    return float(value) if np.isfinite(value) else None


def de_ess(audio: np.ndarray, sr: int, threshold_db: float = -28.0, ratio: float = 4.0,
           split_hz: float = 5500.0) -> np.ndarray:
    """Split-band de-esser: compress only the band above `split_hz` when it gets hot.
    The split is complementary (low = x - high) so untouched audio passes through exactly."""
    from pedalboard import HighpassFilter, Pedalboard

    high = Pedalboard([HighpassFilter(split_hz), HighpassFilter(split_hz)])(audio, sr)
    low = audio - high
    env = np.abs(high).max(axis=0)
    # one-pole envelope follower: ~1 ms attack is approximated by the 5 ms smoothing
    alpha = np.exp(-1.0 / (0.005 * sr))
    env = lfilter([1 - alpha], [1, -alpha], env)
    threshold = db_to_gain(threshold_db)
    over = np.maximum(env / threshold, 1.0)
    gain = over ** (1.0 / ratio - 1.0)
    return (low + high * gain[None, :]).astype(np.float32)


def polish(audio: np.ndarray, sr: int) -> np.ndarray:
    from pedalboard import (Compressor, HighpassFilter, NoiseGate, Pedalboard, PeakFilter)

    # Bring speech to a known level so the gate and compressor thresholds mean the same thing for every file.
    level = loudness(audio, sr)
    if level is not None:
        audio = audio * db_to_gain(-20.0 - level)

    pre = Pedalboard([
        HighpassFilter(80.0),
        NoiseGate(threshold_db=-58.0, ratio=1.5, attack_ms=2.0, release_ms=200.0),
        PeakFilter(cutoff_frequency_hz=300.0, gain_db=-1.5, q=1.0),   # less boxiness
        PeakFilter(cutoff_frequency_hz=4000.0, gain_db=2.0, q=0.7),   # presence
    ])
    audio = pre(audio.astype(np.float32), sr)
    audio = de_ess(audio, sr)
    post = Pedalboard([Compressor(threshold_db=-24.0, ratio=2.5, attack_ms=8.0, release_ms=120.0)])
    return post(audio, sr)


def peak_limit(audio: np.ndarray, sr: int, ceiling_db: float = TRUE_PEAK_DB - 0.5,
               window_ms: float = 8.0) -> np.ndarray:
    """Look-ahead peak limiter with no make-up gain (pedalboard's Limiter adds gain).

    The gain curve is the per-sample gain needed to stay under the ceiling, spread
    with a moving minimum and then smoothed. Every smoothed value is still at or
    below what its own sample needs, so peaks never pass the ceiling."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    ceiling = db_to_gain(ceiling_db)
    peak = np.abs(audio).max(axis=0)
    if peak.max() <= ceiling:
        return audio
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    half = max(1, int(sr * window_ms / 1000))
    gain = minimum_filter1d(need, size=2 * half + 1, mode="nearest")
    gain = uniform_filter1d(gain, size=half + 1, mode="nearest")
    gain = np.minimum(gain, need)  # guard against float rounding at the edges
    return (audio * gain[None, :]).astype(np.float32)


def normalize(audio: np.ndarray, sr: int, target_lufs: float, true_peak_db: float = TRUE_PEAK_DB) -> np.ndarray:
    out = audio
    # Limiting removes peaks that inflated the first measurement, so re-measure and correct.
    for _ in range(3):
        level = loudness(out, sr)
        if level is None or abs(level - target_lufs) < 0.2:
            break
        out = peak_limit((out * db_to_gain(target_lufs - level)).astype(np.float32), sr, true_peak_db - 0.5)
    return out


def match_length(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[-1] >= length:
        return audio[..., :length]
    return np.pad(audio, [(0, 0)] * (audio.ndim - 1) + [(0, length - audio.shape[-1])])


def match_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if audio.shape[0] == channels:
        return audio
    if audio.shape[0] == 1:
        return np.repeat(audio, channels, axis=0)
    mono = audio.mean(axis=0, keepdims=True)
    return np.repeat(mono, channels, axis=0) if channels > 1 else mono


def noise_floor_db(audio: np.ndarray, sr: int, window_s: float = 0.05) -> float:
    """10th-percentile short-term RMS: a simple estimate of the background noise level."""
    mono = audio.mean(axis=0)
    n = max(1, int(window_s * sr))
    frames = mono[: len(mono) // n * n].reshape(-1, n)
    if frames.size == 0:
        return -120.0
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return float(20 * np.log10(np.percentile(rms, 10) + 1e-9))


def peak_db(audio: np.ndarray) -> float:
    return float(20 * np.log10(np.abs(audio).max() + 1e-9))
