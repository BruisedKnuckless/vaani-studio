"""Dub: speak a translation of the audio, in the original speaker's voice or a studio voice.

1. Clean the speech (and separate it from music first, if asked), transcribe it, and pick a
   10–15 s reference clip of the speaker.
2. Translate.
3. Voice it:
   - natural timing (default): whole sentences, spoken at natural speed, laid out one after
     another with the speaker's own pauses. Nothing is sped up or cut.
   - original timing: each line placed where it was spoken, sped up if needed (for video).
4. Finish with EQ, de-essing, gentle compression and loudness normalisation.

Options:
  timing            natural | original
  keep_background   bool   also write a version with the original music/ambience (default false)
  isolate_vocals    bool   separate the voice from music before cloning (for music beds)
  clarity           bool   clarity polish and loudness normalisation (default true)
  voice / voice_name       which voice engine / studio voice (see the registry)
  pace              float  speaking pace, 0.9 relaxed … 1.1 brisk (default 1.0)
  script            list   reviewed segments with "translation" (skips recognition and translation)
  reference         dict   {"start": s, "end": s} to choose the voice sample yourself
  voice_match       bool   extra voice-conversion pass (voices made on this Mac only)
  format            wav | m4a | mp3
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from .. import audio_io
from ..formats import subtitles
from ..providers.base import Segment, Stage, Transcript
from . import dsp
from .transcribe import translate

SR = 48000
MAX_STRETCH = 1.3       # original timing: speed-up applied when a line runs long
SPILL_S = 0.8           # original timing: a line may run this far into the following silence
FADE_S = 0.012
MIN_PAUSE_S, MAX_PAUSE_S = 0.25, 1.0
TARGET_LUFS = -16.0
SENTENCE_END = re.compile(r"[.!?…।][\"'”’)\]]*$")


# ---------------------------------------------------------------- helpers ---

@dataclass
class Passage:
    """Consecutive lines spoken as one piece, so a sentence is never split between voice calls."""
    indices: list[int]
    text: str
    start: float           # where the speech began in the original
    end: float
    gap_after: float = 0.6  # the speaker's pause before the next passage
    original_voice: bool = False  # couldn't be translated: keep the speaker's own audio
    placed_at: float = 0.0
    spoken: float = 0.0
    extra: dict = field(default_factory=dict)


def group_sentences(segments: list[Segment], max_chars: int = 400, max_gap: float = 1.5,
                    keep_original: set[int] | None = None) -> list[Passage]:
    """Merge consecutive translated lines until a sentence ends, the passage gets long, or the
    speaker pauses for a while. Untranslatable lines become their own passages."""
    keep_original = keep_original or set()
    passages: list[Passage] = []
    current: Passage | None = None

    def close():
        nonlocal current
        if current:
            passages.append(current)
            current = None

    for i, seg in enumerate(segments):
        if i in keep_original:
            close()
            passages.append(Passage([i], "", seg.start, seg.end, original_voice=True))
            continue
        text = clean_for_speech(seg.translation or "")
        if not text:
            close()
            continue
        if current is not None:
            paused = seg.start - current.end > max_gap
            too_long = len(current.text) + 1 + len(text) > max_chars
            if paused or too_long:
                close()
        if current is None:
            current = Passage([i], text, seg.start, seg.end)
        else:
            current.indices.append(i)
            current.text = f"{current.text} {text}"
            current.end = seg.end
        if SENTENCE_END.search(current.text):
            close()
    close()
    for a, b in zip(passages, passages[1:]):
        a.gap_after = max(0.0, b.start - a.end)
    return passages


def layout_natural(passages: list[Passage], durations: list[float],
                   min_pause: float = MIN_PAUSE_S, max_pause: float = MAX_PAUSE_S) -> float:
    """Place passages one after another with the speaker's own pauses (clamped).
    Sets each passage's placed_at/spoken and returns when the last one finishes."""
    t = passages[0].start if passages else 0.0
    end = t
    for passage, duration in zip(passages, durations):
        passage.placed_at = t
        passage.spoken = duration
        end = t + duration
        t = end + min(max(passage.gap_after, min_pause), max_pause)
    return end


def pick_reference(segments: list[Segment], audio: np.ndarray | None = None, sr: int = SR,
                   min_s: float = 10.0, max_s: float = 14.5, max_gap: float = 0.8) -> tuple[float, float, str]:
    """The stretch of consecutive speech that best represents the speaker's voice: 10–15 s,
    continuous, with steady energy, and not the closing sign-off (e.g. "thank you")."""
    best: tuple[float, float, float, str] | None = None
    last = len(segments) - 1
    for i in range(len(segments)):
        speech, texts = 0.0, []
        for j in range(i, len(segments)):
            if j > i and segments[j].start - segments[j - 1].end > max_gap:
                break
            span = segments[j].end - segments[i].start
            if span > max_s and j > i:
                break
            speech += segments[j].duration
            texts.append(segments[j].text)
            start, end = segments[i].start, min(segments[j].end, segments[i].start + max_s)
            score = min(speech, end - start)
            if end - start < min_s:
                score -= (min_s - (end - start)) * 1.5
            if j == last and len(segments) > 2:
                score -= 3.0  # sign-offs are delivered differently from the body of a talk
            if audio is not None:
                score -= 4.0 * _energy_unsteadiness(audio, sr, start, end)
            if best is None or score > best[0]:
                best = (score, start, end, " ".join(texts))
    if best is None:
        raise ValueError("No speech found to use as a voice sample")
    return best[1], best[2], best[3]


def _energy_unsteadiness(audio: np.ndarray, sr: int, start: float, end: float) -> float:
    """0 for an even delivery, up to 1 for a clip that swings between loud and quiet."""
    clip = audio_io.as_2d(audio)[0, int(start * sr):int(end * sr)]
    n = int(0.1 * sr)
    if clip.size < n * 4:
        return 0.0
    frames = clip[: clip.size // n * n].reshape(-1, n)
    rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-6)
    voiced = rms_db[rms_db > rms_db.max() - 30]  # ignore the gaps between words
    return float(min(1.0, voiced.std() / 12.0)) if voiced.size > 3 else 0.0


def time_slots(segments: list[Segment], total: float, spill: float = SPILL_S) -> list[float]:
    """Original timing: seconds available for each line, including a little of the silence after it."""
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
    """Original timing: speed speech up (pitch kept) when it's longer than its slot."""
    length = audio.shape[-1] / sr
    ratio = length / slot if slot > 0 else 1.0
    if ratio <= 1.0:
        return audio, 1.0
    from pedalboard import time_stretch

    ratio = min(ratio, max_stretch)
    stretched = time_stretch(audio_io.as_2d(audio).astype(np.float32), sr, stretch_factor=ratio,
                             high_quality=True)
    return stretched, ratio


def place(track: np.ndarray, clip: np.ndarray, sr: int, start: float, max_len: float | None = None) -> None:
    """Add `clip` into `track` at `start` (optionally clipped to `max_len` s), with short fades."""
    clip = audio_io.as_2d(clip)[0].copy()
    if max_len is not None:
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
    natural = opts.get("timing", "natural") != "original"

    ctx.progress(0.01, "Reading audio")
    original = audio_io.load(ctx.input_path, sr=SR)
    total = original.shape[-1] / SR

    # The voice to work from: separated from music only when asked, otherwise the original
    # (separation leaves artefacts that a voice clone would copy).
    background = None
    source = original
    if ctx.has_stage(Stage.SEPARATE):
        stems = ctx.run_stage(Stage.SEPARATE,
                              lambda p: p.separate(original, SR, ctx.span(0.02, 0.16, "Separating voice")))
        source, background = stems["vocals"], stems["background"]
        ctx.release(Stage.SEPARATE)

    ctx.progress(0.17, "Cleaning the voice")
    denoiser = ctx.registry.get("deepfilternet")
    clean, sr = denoiser.denoise(source, SR, ctx.span(0.17, 0.22, "Cleaning the voice"))
    clean = dsp.match_length(audio_io.resample(clean, sr, SR), source.shape[-1])
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
            Stage.ASR, lambda p: p.transcribe(audio16, ctx.src_lang, ctx.span(0.23, 0.42, "Transcribing")))
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
        # Natural timing has no slots to squeeze into, so translate for quality, not brevity.
        untranslated = set(translate(ctx, transcript, tgt, 0.42, 0.55, durations=None if natural else slots))
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

    # Voice sample (used by cloned voices).
    ref = opts.get("reference")
    if ref:
        ref_start, ref_end = float(ref["start"]), float(ref["end"])
        ref_text = " ".join(s.text for s in segments if s.start >= ref_start - 0.1 and s.end <= ref_end + 0.1)
    else:
        ref_start, ref_end, ref_text = pick_reference(segments, clean_mono, SR)
    ref_clip = audio_io.resample(clean_mono[:, int(ref_start * SR):int(ref_end * SR)], SR, 24000)
    ref_clip = dsp.normalize(ref_clip, 24000, -20.0)
    ref_path = audio_io.save(ctx.out_dir / "voice_sample", ref_clip, 24000, "wav")
    ctx.log(f"Voice sample: {ref_start:.1f}–{ref_end:.1f} s")

    tts = ctx.provider_for(Stage.TTS)
    voice_name = opts.get("voice_name")
    pace = float(opts.get("pace", 1.0))

    def speak(text: str, max_duration: float | None = None) -> np.ndarray:
        clip, clip_sr = ctx.run_stage(Stage.TTS, lambda p: p.synthesize(
            text, tgt, str(ref_path), ref_text, max_duration=max_duration if p.kind == "cloud" else None,
            voice=voice_name, pace=pace))
        clip = trim_silence(audio_io.resample(audio_io.to_mono(clip), clip_sr, SR), SR)
        used = ctx.registry.get(ctx.job.providers_used.get(Stage.TTS.value, tts.id))
        if ctx.has_stage(Stage.VOICE_MATCH) and used.kind == "local":
            matched, msr = ctx.run_stage(Stage.VOICE_MATCH, lambda p: p.convert(clip, SR, str(ref_path)))
            clip = trim_silence(audio_io.resample(audio_io.to_mono(matched), msr, SR), SR)
        return clip

    if natural:
        voice, timed_segments, report = _voice_natural(ctx, segments, untranslated, clean_mono, speak,
                                                       max_chars=400 if tts.kind == "cloud" else 220)
    else:
        voice, timed_segments, report = _voice_original(ctx, segments, untranslated, clean_mono, slots,
                                                        speak, translator, transcript.language, tgt, total)
    ctx.job.providers_used[Stage.TTS.value] = ctx.job.providers_used.get(Stage.TTS.value, tts.id)
    ctx.release(Stage.TTS)
    if translator and translator.heavy:
        translator.unload()

    # Finish.
    ctx.progress(0.93, "Polishing the voice")
    if opts.get("clarity", True):
        voice = dsp.normalize(dsp.polish(voice, SR), SR, TARGET_LUFS)
        ctx.job.providers_used[Stage.POLISH.value] = "dsp_polish"
    else:
        target = dsp.loudness(audio_io.to_mono(source), SR) or -20.0
        level = dsp.loudness(voice, SR)
        if level is not None:
            voice = dsp.peak_limit((voice * dsp.db_to_gain(target - level)).astype(np.float32), SR)

    ctx.progress(0.97, "Saving")
    outputs = {
        "dubbed": str(audio_io.save(ctx.out_dir / "dubbed", voice, SR, fmt)),
        "voice_sample": str(ref_path),
        "original": str(ctx.input_path),
    }
    if opts.get("keep_background") and background is not None:
        bed = dsp.match_length(background, voice.shape[-1])
        mix = dsp.peak_limit((dsp.match_channels(voice, bed.shape[0]) + bed).astype(np.float32), SR)
        outputs["dubbed_with_background"] = str(audio_io.save(ctx.out_dir / "dubbed_with_background", mix, SR, fmt))

    srt = ctx.out_dir / f"dubbed_{tgt}.srt"
    srt.write_text(subtitles.to_srt(timed_segments, "translation"), encoding="utf-8")
    outputs["subtitles"] = str(srt)
    script = {"source_language": transcript.language, "target_language": tgt,
              "timing": "natural" if natural else "original",
              "voice_sample": {"start": ref_start, "end": ref_end},
              "segments": [s.to_dict() for s in segments], "voiced": report}
    (ctx.out_dir / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.update(script=str(ctx.out_dir / "script.json"), segments=script["segments"],
                   language=transcript.language, target_language=tgt)
    stretched = sum(1 for r in report if r.get("speed_up", 1.0) > 1.0)
    cut = sum(1 for r in report if r.get("cut"))
    outputs["metrics"] = {"lines": float(len(report)), "sped_up": float(stretched), "cut_short": float(cut),
                          "duration_s": voice.shape[-1] / SR, "original_duration_s": total}
    if cut:
        ctx.warn(f"{cut} line(s) were too long for their time slot and were cut; edit the script to shorten them")
    return outputs


def _voice_natural(ctx, segments, untranslated, clean_mono, speak, max_chars):
    """Whole sentences at natural speed, laid out with the speaker's pauses."""
    passages = group_sentences(segments, max_chars=max_chars, keep_original=untranslated)
    if not passages:
        raise ValueError("Nothing to voice: every line is empty")
    clips = []
    for n, passage in enumerate(passages):
        ctx.progress(0.56 + 0.36 * n / len(passages), f"Voicing passage {n + 1}/{len(passages)}")
        if passage.original_voice:
            clips.append(clean_mono[:, int(passage.start * SR):int(passage.end * SR)])
        else:
            clips.append(speak(passage.text))
    end = layout_natural(passages, [c.shape[-1] / SR for c in clips])
    voice = np.zeros((1, int((end + 0.5) * SR)), dtype=np.float32)
    timed, report = [], []
    for passage, clip in zip(passages, clips):
        place(voice, clip, SR, passage.placed_at)
        source_text = " ".join(segments[i].text for i in passage.indices)
        timed.append(Segment(round(passage.placed_at, 2), round(passage.placed_at + passage.spoken, 2),
                             source_text, passage.text or None))
        report.append({"lines": [i + 1 for i in passage.indices], "at": round(passage.placed_at, 2),
                       "spoken": round(passage.spoken, 2), "original_voice": passage.original_voice,
                       "text": passage.text})
    return voice, timed, report


def _voice_original(ctx, segments, untranslated, clean_mono, slots, speak, translator, src, tgt, total):
    """Each line where it was spoken, sped up (or shortened) to fit its slot. For syncing with video."""
    voice = np.zeros((1, int(total * SR)), dtype=np.float32)
    report = []
    for i in untranslated:
        seg = segments[i]
        place(voice, clean_mono[:, int(seg.start * SR):int(seg.end * SR)], SR, seg.start, slots[i])
        seg.translation = None
    lines = [(i, s) for i, s in enumerate(segments) if clean_for_speech(s.translation or "")]
    can_shorten = bool(translator and translator.follows_length)
    for n, (i, seg) in enumerate(lines):
        ctx.progress(0.56 + 0.36 * n / max(len(lines), 1), f"Voicing line {n + 1}/{len(lines)}")
        text = clean_for_speech(seg.translation)
        clip = speak(text, max_duration=slots[i])
        spoken = clip.shape[-1] / SR
        if can_shorten and spoken / slots[i] > MAX_STRETCH:
            shorter = translator.translate([seg.text], src, tgt, [slots[i] * MAX_STRETCH * 0.85])[0]
            if shorter.strip() and len(shorter) < len(text):
                ctx.log(f"Line {i + 1} was {spoken:.1f}s for a {slots[i]:.1f}s slot; shortened the translation")
                seg.translation = shorter
                clip = speak(clean_for_speech(shorter), max_duration=slots[i])
                spoken = clip.shape[-1] / SR
        clip, ratio = fit_to_slot(clip, SR, slots[i])
        place(voice, clip, SR, seg.start, slots[i])
        report.append({"line": i + 1, "start": seg.start, "slot": round(slots[i], 2), "spoken": round(spoken, 2),
                       "speed_up": round(ratio, 2), "cut": spoken / ratio > slots[i] + 0.05})
    return voice, segments, report
