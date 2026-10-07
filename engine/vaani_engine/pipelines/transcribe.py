"""Transcribe & translate.

Options:
  enhance_first  bool   remove noise before recognising speech
  segments       list   an edited transcript to (re-)translate instead of running speech recognition
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import audio_io
from ..formats import subtitles
from ..providers.base import Segment, Stage, Transcript


def run(ctx) -> dict:
    opts = ctx.options
    if opts.get("segments"):
        transcript = Transcript(ctx.src_lang, [Segment(**{k: s[k] for k in ("start", "end", "text")})
                                               for s in opts["segments"]])
        ctx.progress(0.05, "Using the edited transcript")
    else:
        transcript = recognise(ctx, 0.0, 0.75)

    src = transcript.language
    tgt = ctx.tgt_lang
    for note in transcript.notes:
        ctx.warn(note)
    if ctx.has_stage(Stage.TRANSLATE) and tgt:
        if tgt == src:
            ctx.log(f"The audio is already in {tgt}; skipping translation")
        else:
            bad = translate(ctx, transcript, tgt, 0.75, 0.97)
            if bad:
                ctx.warn(f"{len(bad)} of {len(transcript.segments)} lines may not be translated properly "
                         f"(lines {', '.join(str(i + 1) for i in bad)}). Check the transcript for misheard words.")

    ctx.progress(0.98, "Saving subtitles")
    return write_outputs(ctx.out_dir, transcript, tgt if any(s.translation for s in transcript.segments) else None)


def recognise(ctx, start: float, end: float) -> Transcript:
    """Load audio, optionally clean it, and run the planned speech recogniser."""
    ctx.progress(start + 0.01, "Reading audio")
    audio = audio_io.load(ctx.input_path, sr=16000, channels=1)
    if ctx.options.get("enhance_first"):
        denoiser = ctx.registry.get("deepfilternet")
        cleaned, sr = denoiser.denoise(audio_io.resample(audio, 16000, 48000), 48000,
                                       ctx.span(start + 0.02, start + 0.08, "Cleaning audio"))
        audio = audio_io.resample(cleaned, sr, 16000)
        ctx.job.providers_used["denoise"] = denoiser.id
    mono = audio.reshape(-1)
    transcript = ctx.run_stage(
        Stage.ASR, lambda p: p.transcribe(mono, ctx.src_lang, ctx.span(start + 0.08, end, "Transcribing")))
    if ctx.src_lang == "auto":
        ctx.log(f"Detected language: {transcript.language}")
    elif transcript.language == "auto":
        transcript.language = ctx.src_lang
    return transcript


def translate(ctx, transcript: Transcript, tgt: str, start: float, end: float,
              durations: list[float] | None = None) -> list[int]:
    """Translate every segment. Returns the indexes of lines that still aren't in the
    target language after one retry (usually because the transcript itself was wrong)."""
    from ..providers.translation_prompt import clean_output, is_in_language

    texts = [s.text for s in transcript.segments]
    if not texts:
        return []
    src = transcript.language
    translations = ctx.run_stage(
        Stage.TRANSLATE,
        lambda p: p.translate(texts, src, tgt, durations, ctx.span(start, end, "Translating")))
    translations = [clean_output(t) for t in translations]
    bad = [i for i, t in enumerate(translations) if texts[i].strip() and not is_in_language(t, tgt)]
    if bad:
        ctx.log(f"{len(bad)} line(s) came back in the wrong language; retrying them one by one")
        translator = ctx.registry.get(ctx.job.providers_used[Stage.TRANSLATE.value])
        for i in bad:
            retry = translator.translate([texts[i]], src, tgt, [durations[i]] if durations else None)[0]
            if is_in_language(retry, tgt):
                translations[i] = retry
        bad = [i for i in bad if not is_in_language(translations[i], tgt)]
    for seg, text in zip(transcript.segments, translations):
        seg.translation = text
    return bad


def write_outputs(out_dir: Path, transcript: Transcript, tgt: str | None) -> dict:
    out_dir = Path(out_dir)
    segs = transcript.segments
    files = {}
    for ext, writer in (("srt", subtitles.to_srt), ("vtt", subtitles.to_vtt), ("txt", subtitles.to_txt)):
        path = out_dir / f"transcript.{ext}"
        path.write_text(writer(segs, "text"), encoding="utf-8")
        files[f"transcript_{ext}"] = str(path)
        if tgt:
            path = out_dir / f"translation_{tgt}.{ext}"
            path.write_text(writer(segs, "translation"), encoding="utf-8")
            files[f"translation_{ext}"] = str(path)
    data = {"language": transcript.language, "target_language": tgt,
            "segments": [s.to_dict() for s in segs]}
    json_path = out_dir / "transcript.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    files["transcript_json"] = str(json_path)
    return files | {"language": transcript.language, "target_language": tgt,
                    "segments": data["segments"]}
