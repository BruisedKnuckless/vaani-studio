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
from pathlib import Path

from .config import ensure_dirs, settings

from .providers.registry import QUALITY_DENOISERS as DENOISERS


def _mode_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=["local", "cloud", "auto"], default="auto",
                   help="where to process: on this Mac, cloud APIs, or automatic")
    p.add_argument("--offline", action="store_true", help="never use the network")
    p.add_argument("--via-app", action="store_true",
                   help="run on the open Vaani app's engine (uses the API keys saved in the app)")
    for stage in ("separate", "denoise", "asr", "translate", "tts", "voice_match"):
        p.add_argument(f"--{stage.replace('_', '-')}", dest=f"ov_{stage}", metavar="PROVIDER",
                       help=f"force a provider for the {stage} step")


def _overrides(args) -> dict:
    return {k[3:]: v for k, v in vars(args).items() if k.startswith("ov_") and v}


def _run(job_type: str, input_path: str, params: dict, via_app: bool = False) -> int:
    if via_app:
        return _run_via_app(job_type, input_path, params)
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
    _print_plan(job.plan)
    last = ""
    while job.status in ("queued", "running"):
        time.sleep(0.25)
        last = _print_progress(job.to_dict(), last)
    return _print_result(job.to_dict())


def _run_via_app(job_type: str, input_path: str, params: dict) -> int:
    """Run the job on the engine of the open Vaani app, which holds the API keys from the Keychain."""
    import httpx

    from .config import SUPPORT_DIR

    info_file = SUPPORT_DIR / "engine.json"
    if not info_file.exists():
        print("✗ The Vaani app isn't running (open it first).", file=sys.stderr)
        return 2
    info = json.loads(info_file.read_text())
    base = f"http://127.0.0.1:{info['port']}"
    headers = {"Authorization": f"Bearer {info['token']}"}
    body = {"job_type": job_type, "input_path": str(Path(input_path).expanduser().resolve()), **params}
    response = httpx.post(f"{base}/jobs", json=body, headers=headers, timeout=60)
    if response.status_code >= 400:
        print(f"✗ {response.json().get('detail', response.text)}", file=sys.stderr)
        return 2
    job = response.json()
    _print_plan(job["plan"])
    last = ""
    while job["status"] in ("queued", "running"):
        time.sleep(1.0)
        job = httpx.get(f"{base}/jobs/{job['id']}", headers=headers, timeout=60).json()
        last = _print_progress(job, last)
    return _print_result(job)


def _print_plan(plan: dict) -> None:
    print(f"Plan: {plan['summary']}")
    if plan["uploads"]:
        print("⚠ Audio or text will be sent to a cloud API.")


def _print_progress(job: dict, last: str) -> str:
    line = f"[{job['progress'] * 100:5.1f}%] {job['message']}"
    if line != last:
        print(line, flush=True)
    return line


def _print_result(job: dict) -> int:
    if job["status"] != "done":
        print(f"✗ {job['status']}: {job['message']}", file=sys.stderr)
        error = job.get("error") or {}
        if error.get("local_alternative"):
            print(f"  Try: --{error['stage'].replace('_', '-')} {error['local_alternative']}", file=sys.stderr)
        for line in job["log"][-5:]:
            print("  " + line, file=sys.stderr)
        return 1
    print(f"✓ Done in {job['updated_at'] - job['created_at']:.1f}s → {job['out_dir']}")
    print(f"  Used: {json.dumps(job['providers_used'])}")
    for key, value in job["outputs"].items():
        if key == "metrics":
            print("  Metrics: " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in value.items()))
        elif key == "warnings":
            for warning in value:
                print(f"  ⚠ {warning}")
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
    p.add_argument("--voice", choices=["auto", "sarvam_clone", "local", "sarvam_studio"], default="auto",
                   help="speaker's voice via Sarvam / on this Mac, or a Sarvam studio voice")
    p.add_argument("--voice-name", default="shubh", help="Sarvam studio voice, e.g. shubh, ishita")
    p.add_argument("--timing", choices=["natural", "original"], default="natural",
                   help="natural pacing, or keep each line at its original time (for video)")
    p.add_argument("--pace", type=float, default=1.0, help="speaking pace: 0.9 relaxed, 1.0 natural, 1.1 brisk")
    p.add_argument("--with-background", action="store_true", help="also save a version with the original music")
    p.add_argument("--no-clarity", action="store_true", help="skip the clarity polish and loudness levelling")
    p.add_argument("--match-voice", action="store_true", help="extra voice-conversion pass (on-device voices)")
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
        return _run("enhance", args.input, params, args.via_app)
    if args.command == "dub":
        params = {"mode": args.mode, "overrides": _overrides(args), "src_lang": args.src, "tgt_lang": args.tgt,
                  "options": {"voice": args.voice, "voice_name": args.voice_name, "timing": args.timing, "pace": args.pace,
                              "keep_background": args.with_background, "clarity": not args.no_clarity,
                              "voice_match": args.match_voice, "format": args.format}}
        return _run("dub", args.input, params, args.via_app)

    if args.command == "transcribe":
        params = {"mode": args.mode, "overrides": _overrides(args), "src_lang": args.lang,
                  "tgt_lang": args.tgt, "options": {"enhance_first": args.clean}}
        code = _run("transcribe", args.input, params, args.via_app)
        return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
