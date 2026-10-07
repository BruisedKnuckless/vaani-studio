"""Enhance: remove background noise, clarify speech, normalise loudness.

Options:
  isolate_vocals  bool   run voice/background separation first (for music beds)
  strength        0..1   blend of cleaned and original audio (1 = fully cleaned)
  polish          bool   EQ, de-esser and compression
  loudness        podcast | social | broadcast | off
  format          wav | m4a | mp3 | flac
"""

from __future__ import annotations

import json

import numpy as np

from .. import audio_io
from ..providers.base import Stage
from . import dsp

SR = 48000


def run(ctx) -> dict:
    opts = ctx.options
    strength = float(opts.get("strength", 1.0))
    loudness_preset = opts.get("loudness", "podcast")
    fmt = opts.get("format", "wav")

    ctx.progress(0.01, "Reading audio")
    original = audio_io.load(ctx.input_path, sr=SR)
    audio = original
    metrics = {"input_lufs": dsp.loudness(original, SR), "input_noise_floor_db": dsp.noise_floor_db(original, SR)}
    outputs: dict = {}

    denoise_start = 0.05
    if ctx.has_stage(Stage.SEPARATE):
        stems = ctx.run_stage(Stage.SEPARATE, lambda p: p.separate(audio, SR, ctx.span(0.05, 0.45, "Separating")))
        audio = stems["vocals"]
        outputs["background"] = str(audio_io.save(ctx.out_dir / "background", stems["background"], SR, fmt))
        denoise_start = 0.45

    cleaned, sr = ctx.run_stage(Stage.DENOISE,
                                lambda p: p.denoise(audio, SR, ctx.span(denoise_start, 0.85, "Removing noise")))
    cleaned = audio_io.resample(cleaned, sr, SR)
    cleaned = dsp.match_channels(dsp.match_length(cleaned, original.shape[-1]), original.shape[0])

    if strength < 1.0:
        dry = audio if ctx.has_stage(Stage.SEPARATE) else original
        cleaned = strength * cleaned + (1.0 - strength) * dry

    ctx.check_cancel()
    if opts.get("polish", True):
        ctx.progress(0.88, "Polishing voice")
        cleaned = dsp.polish(cleaned, SR)
        ctx.job.providers_used[Stage.POLISH.value] = "dsp_polish"

    if loudness_preset in dsp.LOUDNESS_PRESETS:
        ctx.progress(0.94, "Normalising loudness")
        cleaned = dsp.normalize(cleaned, SR, dsp.LOUDNESS_PRESETS[loudness_preset])
    else:
        peak = np.abs(cleaned).max()
        if peak > 0.99:
            cleaned = cleaned * (0.99 / peak)

    ctx.progress(0.97, "Saving")
    out_path = audio_io.save(ctx.out_dir / "enhanced", cleaned, SR, fmt)
    outputs["enhanced"] = str(out_path)
    outputs["original"] = str(ctx.input_path)

    metrics.update({
        "output_lufs": dsp.loudness(cleaned, SR),
        "output_noise_floor_db": dsp.noise_floor_db(cleaned, SR),
        "output_peak_db": dsp.peak_db(cleaned),
        "duration_s": original.shape[-1] / SR,
    })
    outputs["metrics"] = metrics
    (ctx.out_dir / "report.json").write_text(json.dumps({"providers": ctx.job.providers_used,
                                                          "metrics": metrics}, indent=2))
    ctx.log(f"Noise floor {metrics['input_noise_floor_db']:.1f} → {metrics['output_noise_floor_db']:.1f} dB")
    return outputs
