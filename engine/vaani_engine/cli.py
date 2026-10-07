"""Command-line access to every pipeline, mainly for testing without the app.

    vaani enhance "Timeline 1.wav" --quality high --mode local
    vaani plan dub --from hi --to en --mode auto
    vaani models list
    vaani serve
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .config import ensure_dirs, settings

from .providers.registry import QUALITY_DENOISERS as DENOISERS


def _mode_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=["local", "cloud", "auto"], default="auto",
                   help="where to process: on this Mac, cloud APIs, or automatic")
    p.add_argument("--offline", action="store_true", help="never use the network")
    for stage in ("separate", "denoise", "asr", "translate", "tts", "voice_match"):
        p.add_argument(f"--{stage.replace('_', '-')}", dest=f"ov_{stage}", metavar="PROVIDER",
                       help=f"force a provider for the {stage} step")


def _overrides(args) -> dict:
    return {k[3:]: v for k, v in vars(args).items() if k.startswith("ov_") and v}


def _run(job_type: str, input_path: str, params: dict) -> int:
    from .jobs import JobManager
    from .pipelines import get_pipelines
    from .providers import build_registry

    ensure_dirs()
    manager = JobManager(build_registry(settings), get_pipelines())
    try:
        job = manager.submit(job_type, input_path, params)
    except (PermissionError, FileNotFoundError, ValueError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2
    print(f"Plan: {job.plan['summary']}")
    if job.plan["uploads"]:
        print("⚠ Audio or text will be sent to a cloud API.")
    last = ""
    while job.status in ("queued", "running"):
        time.sleep(0.25)
        line = f"[{job.progress * 100:5.1f}%] {job.message}"
        if line != last:
            print(line, flush=True)
            last = line
    if job.status != "done":
        print(f"✗ {job.status}: {job.message}", file=sys.stderr)
        if job.error and job.error.get("local_alternative"):
            print(f"  Try: --{job.error['stage'].replace('_', '-')} {job.error['local_alternative']}", file=sys.stderr)
        for line in job.log[-5:]:
            print("  " + line, file=sys.stderr)
        return 1
    print(f"✓ Done in {job.updated_at - job.created_at:.1f}s → {job.out_dir}")
    print(f"  Used: {json.dumps(job.providers_used)}")
    for key, value in job.outputs.items():
        if key == "metrics":
            print("  Metrics: " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in value.items()))
        elif key == "segments":
            for seg in value[:12]:
                line = f"  [{seg['start']:6.1f}–{seg['end']:6.1f}] {seg['text']}"
                print(line + (f"\n{'':19}→ {seg['translation']}" if seg.get("translation") else ""))
            if len(value) > 12:
                print(f"  … {len(value) - 12} more segments")
        elif isinstance(value, str):
            print(f"  {key}: {value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vaani")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("enhance", help="remove noise and polish speech")
    p.add_argument("input")
    p.add_argument("--quality", choices=list(DENOISERS), default=None,
                   help="fast = DeepFilterNet, high = MossFormer2, basic = DSP only")
    p.add_argument("--isolate", action="store_true", help="separate voice from music/background first")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--no-polish", action="store_true")
    p.add_argument("--loudness", choices=["podcast", "social", "broadcast", "off"], default="podcast")
    p.add_argument("--format", choices=["wav", "m4a", "mp3", "flac"], default="wav")
    _mode_args(p)

    p = sub.add_parser("transcribe", help="transcribe and optionally translate")
    p.add_argument("input")
    p.add_argument("--lang", default="auto", choices=["auto", "en", "hi", "mr", "hinglish"])
    p.add_argument("--to", dest="tgt", default=None, choices=["en", "hi", "mr", "hinglish"])
    p.add_argument("--clean", action="store_true", help="remove noise before transcribing")
    _mode_args(p)

    p = sub.add_parser("dub", help="dub into another language in the speaker's voice")
    p.add_argument("input")
    p.add_argument("--from", dest="src", default="auto", choices=["auto", "en", "hi", "mr"])
    p.add_argument("--to", dest="tgt", required=True, choices=["en", "hi", "mr"])
    p.add_argument("--match-voice", action="store_true", help="extra voice-conversion pass")
    p.add_argument("--no-background", action="store_true", help="voice only, without the original music/ambience")
    p.add_argument("--format", choices=["wav", "m4a", "mp3"], default="wav")
    _mode_args(p)

    p = sub.add_parser("plan", help="show which provider would run each step")
    p.add_argument("job_type", choices=["enhance", "transcribe", "dub"])
    p.add_argument("--from", dest="src", default="auto")
    p.add_argument("--to", dest="tgt", default=None)
    p.add_argument("--isolate", action="store_true")
    p.add_argument("--match-voice", action="store_true", help="add the voice-match step")
    _mode_args(p)

    p = sub.add_parser("models", help="list or download models")
    p.add_argument("action", choices=["list", "download", "delete"])
    p.add_argument("model_id", nargs="?")

    sub.add_parser("serve", help="start the HTTP engine for the Mac app")

    args = parser.parse_args(argv)
    if getattr(args, "offline", False):
        settings.update(offline=True)

    if args.command == "serve":
        from .server import main as serve

        serve([])
        return 0

    if args.command == "models":
        from .models_manager import models

        if args.action == "list":
            for m in models.list():
                print(f"{m['id']:24} {m['state']:12} {m['size_mb']:>6} MB  {m['license']:28} {m['purpose']}")
            return 0
        if not args.model_id:
            print("model_id is required", file=sys.stderr)
            return 2
        if args.action == "download":
            print(f"Downloading {args.model_id}…")
            print(models.download(args.model_id))
        else:
            models.delete(args.model_id)
        return 0

    if args.command == "plan":
        from .providers import build_registry

        registry = build_registry(settings)
        options = {"isolate_vocals": args.isolate, "voice_match": args.match_voice}
        plan = registry.resolve_plan(args.job_type, args.src, args.tgt, args.mode, _overrides(args), options)
        for step in plan.steps:
            flag = "⛔" if step.blocking else ("☁️ " if step.kind == "cloud" else "💻")
            extra = f"  ({step.note})" if step.note else ""
            fb = f"  fallback: {step.fallback_id}" if step.fallback_id else ""
            print(f"{flag} {step.label:20} {step.provider_name:34} [{step.status}]{extra}{fb}")
        print(f"Uploads to cloud: {'yes' if plan.uploads else 'no'} · Blocked: {'yes' if plan.blocking else 'no'}")
        return 1 if plan.blocking else 0

    if args.command == "enhance":
        params = {
            "mode": args.mode, "overrides": _overrides(args), "src_lang": "auto", "tgt_lang": None,
            "options": {"quality": args.quality or "fast", "isolate_vocals": args.isolate, "strength": args.strength,
                        "polish": not args.no_polish, "loudness": args.loudness, "format": args.format},
        }
        return _run("enhance", args.input, params)
    if args.command == "dub":
        params = {"mode": args.mode, "overrides": _overrides(args), "src_lang": args.src, "tgt_lang": args.tgt,
                  "options": {"voice_match": args.match_voice, "keep_background": not args.no_background,
                              "format": args.format}}
        return _run("dub", args.input, params)

    if args.command == "transcribe":
        params = {"mode": args.mode, "overrides": _overrides(args), "src_lang": args.lang,
                  "tgt_lang": args.tgt, "options": {"enhance_first": args.clean}}
        code = _run("transcribe", args.input, params)
        return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
