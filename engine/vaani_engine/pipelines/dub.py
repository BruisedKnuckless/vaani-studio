"""Dub: speak a translation of the audio in the original speaker's voice.

1. Separate the voice from music/background (kept for the final mix).
2. Clean the voice, transcribe it, and pick a 6–14 s reference clip of the speaker.
3. Translate line by line with a time budget per line.
4. Generate each line in the cloned voice, fit it into its time slot, and place it.
5. Mix the new voice over the original background at the original loudness.

Options:
  script            list   reviewed segments with "translation" (skips recognition and translation)
  reference         dict   {"start": s, "end": s} to choose the voice sample yourself
  keep_background   bool   mix the original music/ambience back in (default true)
  voice_match       bool   extra voice-conversion pass toward the original speaker
  format            wav | m4a | mp3
"""

from __future__ import annotations

import json
import re

import numpy as np

from .. import audio_io
from ..formats import subtitles
from ..providers.base import Segment, Stage, Transcript
from . import dsp
from .transcribe import translate

SR = 48000
MAX_STRETCH = 1.3       # speed-up applied silently when a line runs long
SPILL_S = 0.8           # a line may run this far into the following silence
FADE_S = 0.012


# ---------------------------------------------------------------- helpers ---

def pick_reference(segments: list[Segment], min_s: float = 6.0, max_s: float = 14.0,
                   max_gap: float = 0.8) -> tuple[float, float, str]:
    """The stretch of consecutive speech that best represents the speaker's voice."""
    best: tuple[float, float, float, str] | None = None
    for i in range(len(segments)):
        speech, texts = 0.0, []
        for j in range(i, len(segments)):
            if j > i and segments[j].start - segments[j - 1].end > max_gap:
                break
            span = segments[j].end - segments[i].start
            if span > max_s:
                break
            speech += segments[j].duration
            texts.append(segments[j].text)
            score = speech - (0 if span >= min_s else 5.0)
            if best is None or score > best[0]:
                best = (score, segments[i].start, segments[j].end, " ".join(texts))
    if best is None:
        raise ValueError("No speech found to use as a voice sample")
    _, start, end, text = best
    if end - start > max_s:
        end = start + max_s
    return start, end, text


def time_slots(segments: list[Segment], total: float, spill: float = SPILL_S) -> list[float]:
    """Seconds available for each line: its own length plus a little of the silence after it."""
    slots = []
    for i, seg in enumerate(segments):
        limit = segments[i + 1].start - 0.05 if i + 1 < len(segments) else total
        slots.append(max(seg.duration, min(seg.end + spill, limit) - seg.start))
    return slots


def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -42.0, pad_s: float = 0.03) -> np.ndarray:
    mono = np.abs(audio_io.as_2d(audio)).max(axis=0)
    if mono.size == 0 or mono.max() <= 0:
        return audio
    level = mono.max() * dsp.db_to_gain(threshold_db)
    loud = np.flatnonzero(mono > level)
    pad = int(pad_s * sr)
    start, end = max(0, loud[0] - pad), min(mono.size, loud[-1] + pad + 1)
    return audio_io.as_2d(audio)[:, start:end]


def fit_to_slot(audio: np.ndarray, sr: int, slot: float, max_stretch: float = MAX_STRETCH) -> tuple[np.ndarray, float]:
    """Speed speech up (pitch kept) when it's longer than its slot. Returns (audio, ratio applied)."""
    length = audio.shape[-1] / sr
    ratio = length / slot if slot > 0 else 1.0
    if ratio <= 1.0:
        return audio, 1.0
    from pedalboard import time_stretch

    ratio = min(ratio, max_stretch)
    stretched = time_stretch(audio_io.as_2d(audio).astype(np.float32), sr, stretch_factor=ratio,
                             high_quality=True)
    return stretched, ratio


def place(track: np.ndarray, clip: np.ndarray, sr: int, start: float, max_len: float) -> None:
    """Add `clip` into `track` at `start`, clipped to `max_len` seconds, with short fades."""
    clip = audio_io.as_2d(clip)[0].copy()
    clip = clip[: int(max_len * sr)]
    fade = min(int(FADE_S * sr), clip.size // 2)
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        clip[:fade] *= ramp
        clip[-fade:] *= ramp[::-1]
    a = int(start * sr)
    b = min(track.shape[-1], a + clip.size)
    track[0, a:b] += clip[: b - a]


def clean_for_speech(text: str) -> str:
    text = re.sub(r"[\[\(\{].*?[\]\)\}]", "", text or "")
    return " ".join(text.replace("*", "").split())


# ---------------------------------------------------------------- pipeline ---

def run(ctx) -> dict:
    opts = ctx.options
    tgt = ctx.tgt_lang
    if not tgt or tgt not in ("en", "hi", "mr"):
        raise ValueError("Choose English, Hindi or Marathi as the dubbing language")
    fmt = opts.get("format", "wav")

    ctx.progress(0.01, "Reading audio")
    original = audio_io.load(ctx.input_path, sr=SR)
    total = original.shape[-1] / SR

    stems = ctx.run_stage(Stage.SEPARATE,
                          lambda p: p.separate(original, SR, ctx.span(0.02, 0.18, "Separating voice")))
    vocals, background = stems["vocals"], stems["background"]
    ctx.release(Stage.SEPARATE)

    ctx.progress(0.19, "Cleaning the voice")
    denoiser = ctx.registry.get("deepfilternet")
    clean, sr = denoiser.denoise(vocals, SR, ctx.span(0.19, 0.24, "Cleaning the voice"))
    clean = dsp.match_length(audio_io.resample(clean, sr, SR), vocals.shape[-1])
    ctx.job.providers_used["denoise"] = denoiser.id
    clean_mono = audio_io.to_mono(clean)

    # Script: from a reviewed transcript, or recognise + translate now.
    if opts.get("script"):
        segments = [Segment(s["start"], s["end"], s.get("text", ""), s.get("translation"))
                    for s in opts["script"]]
        transcript = Transcript(ctx.src_lang, segments)
        ctx.log(f"Using the reviewed script ({len(segments)} lines)")
    else:
        audio16 = audio_io.resample(clean_mono, SR, 16000).reshape(-1)
        transcript = ctx.run_stage(
            Stage.ASR, lambda p: p.transcribe(audio16, ctx.src_lang, ctx.span(0.25, 0.45, "Transcribing")))
        segments = transcript.segments
        ctx.release(Stage.ASR)
        if not segments:
            raise ValueError("No speech was found to dub")
        if transcript.language == "auto":
            transcript.language = ctx.src_lang
        ctx.log(f"Source language: {transcript.language}; {len(segments)} lines")

    slots = time_slots(segments, total)
    for note in transcript.notes:
        ctx.warn(note)
    untranslated: set[int] = set()
    if not opts.get("script"):
        if transcript.language == tgt:
            raise ValueError("The audio is already in the dubbing language")
        untranslated = set(translate(ctx, transcript, tgt, 0.45, 0.55, durations=slots))
        if len(untranslated) > 0.4 * len(segments):
            raise ValueError(
                f"Only {len(segments) - len(untranslated)} of {len(segments)} lines could be translated into "
                f"{'English' if tgt == 'en' else tgt}, which almost always means the speech was misheard. "
                + ("Whisper struggles with Marathi: dub with Cloud (Sarvam), or add a free Hugging Face token so "
                   "Vaani can use IndicConformer. " if transcript.language == "mr" else "")
                + "You can also fix the transcript on the Transcribe screen and dub from there.")
        if untranslated:
            ctx.warn(f"{len(untranslated)} line(s) couldn't be translated and keep the original voice "
                     f"(lines {', '.join(str(i + 1) for i in sorted(untranslated))})")
    translator = ctx.registry.get(ctx.job.providers_used["translate"]) if "translate" in ctx.job.providers_used else None
    can_shorten = bool(translator and translator.follows_length)

    # Voice sample.
    ref = opts.get("reference")
    if ref:
        ref_start, ref_end = float(ref["start"]), float(ref["end"])
        ref_text = " ".join(s.text for s in segments if s.start >= ref_start - 0.1 and s.end <= ref_end + 0.1)
    else:
        ref_start, ref_end, ref_text = pick_reference(segments)
    ref_clip = audio_io.resample(clean_mono[:, int(ref_start * SR):int(ref_end * SR)], SR, 24000)
    ref_clip = dsp.normalize(ref_clip, 24000, -20.0)
    ref_path = audio_io.save(ctx.out_dir / "voice_sample", ref_clip, 24000, "wav")
    ctx.log(f"Voice sample: {ref_start:.1f}–{ref_end:.1f} s")

    # Speak each line in the cloned voice.
    voice = np.zeros((1, original.shape[-1]), dtype=np.float32)
    report = []
    for i in untranslated:
        seg = segments[i]
        place(voice, clean_mono[:, int(seg.start * SR):int(seg.end * SR)], SR, seg.start, slots[i])
        seg.translation = None
    lines = [(i, s) for i, s in enumerate(segments) if clean_for_speech(s.translation or "")]
    tts = ctx.provider_for(Stage.TTS)
    for n, (i, seg) in enumerate(lines):
        ctx.progress(0.56 + 0.36 * n / max(len(lines), 1), f"Voicing line {n + 1}/{len(lines)}")
        text = clean_for_speech(seg.translation)
        clip, clip_sr = ctx.run_stage(Stage.TTS, lambda p: p.synthesize(
            text, tgt, str(ref_path), ref_text, max_duration=slots[i] if p.kind == "cloud" else None, voice=opts.get("voice_name")))
        clip = trim_silence(audio_io.resample(audio_io.to_mono(clip), clip_sr, SR), SR)
        voiced_locally = ctx.registry.get(ctx.job.providers_used.get(Stage.TTS.value, tts.id)).kind == "local"
        if ctx.has_stage(Stage.VOICE_MATCH) and voiced_locally:
            matched, msr = ctx.run_stage(Stage.VOICE_MATCH, lambda p: p.convert(clip, SR, str(ref_path)))
            clip = trim_silence(audio_io.resample(audio_io.to_mono(matched), msr, SR), SR)
        natural = clip.shape[-1] / SR
        if can_shorten and natural / slots[i] > MAX_STRETCH:
            # Too long even after speeding up: ask for a shorter line and voice it again.
            budget = slots[i] * MAX_STRETCH * 0.85
            shorter = translator.translate([seg.text], transcript.language, tgt, [budget])[0]
            if shorter.strip() and len(shorter) < len(text):
                ctx.log(f"Line {i + 1} was {natural:.1f}s for a {slots[i]:.1f}s slot; shortened the translation")
                seg.translation = shorter
                retry, rsr = ctx.run_stage(Stage.TTS, lambda p: p.synthesize(
                    clean_for_speech(shorter), tgt, str(ref_path), ref_text,
                    max_duration=slots[i] if p.kind == "cloud" else None, voice=opts.get("voice_name")))
                clip = trim_silence(audio_io.resample(audio_io.to_mono(retry), rsr, SR), SR)
                natural = clip.shape[-1] / SR
        clip, ratio = fit_to_slot(clip, SR, slots[i])
        place(voice, clip, SR, seg.start, slots[i])
        report.append({"line": i + 1, "start": seg.start, "slot": round(slots[i], 2),
                       "spoken": round(natural, 2), "speed_up": round(ratio, 2),
                       "cut": natural / ratio > slots[i] + 0.05})
    ctx.job.providers_used[Stage.TTS.value] = ctx.job.providers_used.get(Stage.TTS.value, tts.id)

    ctx.release(Stage.TTS)
    if translator and translator.heavy:
        translator.unload()

    # Mix.
    ctx.progress(0.93, "Mixing")
    target = dsp.loudness(audio_io.to_mono(vocals), SR) or -20.0
    level = dsp.loudness(voice, SR)
    if level is not None:
        voice = voice * dsp.db_to_gain(target - level)
    voice_out = dsp.match_channels(voice, original.shape[0])
    mix = voice_out + background if opts.get("keep_background", True) else voice_out
    mix = dsp.peak_limit(mix.astype(np.float32), SR)

    ctx.progress(0.97, "Saving")
    outputs = {
        "dubbed": str(audio_io.save(ctx.out_dir / "dubbed", mix, SR, fmt)),
        "dubbed_voice": str(audio_io.save(ctx.out_dir / "dubbed_voice_only", voice_out, SR, "wav")),
        "background": str(audio_io.save(ctx.out_dir / "background", background, SR, "wav")),
        "voice_sample": str(ref_path),
        "original": str(ctx.input_path),
    }
    (ctx.out_dir / f"dubbed_{tgt}.srt").write_text(subtitles.to_srt(segments, "translation"), encoding="utf-8")
    outputs["subtitles"] = str(ctx.out_dir / f"dubbed_{tgt}.srt")
    script = {"source_language": transcript.language, "target_language": tgt,
              "voice_sample": {"start": ref_start, "end": ref_end},
              "segments": [s.to_dict() for s in segments], "timing": report}
    (ctx.out_dir / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["script"] = str(ctx.out_dir / "script.json")
    outputs["segments"] = script["segments"]
    outputs["language"] = transcript.language
    outputs["target_language"] = tgt
    cut = sum(1 for r in report if r["cut"])
    stretched = sum(1 for r in report if r["speed_up"] > 1.0)
    outputs["metrics"] = {"lines": float(len(report)), "sped_up": float(stretched), "cut_short": float(cut),
                          "duration_s": total}
    if cut:
        ctx.log(f"{cut} line(s) were still too long after speeding up and were cut; edit the script to shorten them")
    return outputs
