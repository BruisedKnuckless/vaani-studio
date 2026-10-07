"""HTTP API used by the Mac app. Listens on 127.0.0.1 only and requires the
bearer token the app generates at launch (VAANI_TOKEN)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import socket
import sys
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__, audio_io
from .config import ensure_dirs, settings
from .jobs import JobManager
from .models_manager import CATALOG, models
from .pipelines import get_pipelines
from .providers import build_registry
from .providers.base import LANGUAGES

TOKEN = os.environ.get("VAANI_TOKEN") or ""

app = FastAPI(title="Vaani Engine", version=__version__)
registry = build_registry(settings)
manager: JobManager | None = None


def get_manager() -> JobManager:
    global manager
    if manager is None:
        ensure_dirs()
        manager = JobManager(registry, get_pipelines())
    return manager


@app.middleware("http")
async def check_token(request: Request, call_next):
    if TOKEN and request.url.path != "/health":
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(supplied, TOKEN):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


# ---------------------------------------------------------------- models ---

class SettingsIn(BaseModel):
    keys: dict[str, str | None] | None = None
    offline: bool | None = None
    output_dir: str | None = None


class PlanIn(BaseModel):
    job_type: Literal["enhance", "transcribe", "dub"]
    src_lang: str = "auto"
    tgt_lang: str | None = None
    mode: Literal["local", "cloud", "auto"] = "auto"
    overrides: dict[str, str] = Field(default_factory=dict)
    options: dict = Field(default_factory=dict)


class JobIn(PlanIn):
    input_path: str


# ---------------------------------------------------------------- routes ---

@app.get("/health")
def health():
    return {
        "status": "ok", "version": __version__, "offline": settings.offline,
        "keys": settings.key_status(), "output_dir": str(settings.output_dir),
        "languages": LANGUAGES,
    }


@app.post("/settings")
def update_settings(body: SettingsIn):
    settings.update(body.keys, body.offline, body.output_dir)
    ensure_dirs()
    return health()


@app.get("/providers")
def list_providers():
    return [p.describe() | {"status": registry.status_of(p)[0], "status_note": registry.status_of(p)[1]}
            for p in registry.all()]


@app.post("/plan/preview")
def plan_preview(body: PlanIn):
    plan = registry.resolve_plan(body.job_type, body.src_lang, body.tgt_lang, body.mode,
                                 body.overrides, body.options)
    return plan.to_dict() | {"cloud_available": registry.cloud_available_for(body.job_type)}


@app.post("/jobs")
def create_job(body: JobIn):
    params = body.model_dump(exclude={"input_path", "job_type"})
    try:
        job = get_manager().submit(body.job_type, body.input_path, params)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except PermissionError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return job.to_dict()


@app.get("/jobs")
def list_jobs():
    return [j.to_dict() for j in get_manager().list()]


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = get_manager().jobs.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    return job.to_dict()


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if job_id not in get_manager().jobs:
        raise HTTPException(404, "No such job")
    return get_manager().cancel(job_id).to_dict()


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    get_manager().delete(job_id)
    return {"deleted": job_id}


@app.get("/events")
async def events(request: Request):
    """Server-sent events: one `job` event per job update."""
    mgr = get_manager()
    q = mgr.subscribe()

    async def stream():
        try:
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"event: job\ndata: {json.dumps(payload, default=str)}\n\n"
        finally:
            mgr.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


class AudioIn(BaseModel):
    path: str
    buckets: int = 1200


@app.post("/audio/peaks")
def audio_peaks(body: AudioIn):
    p = Path(body.path).expanduser()
    if not p.exists():
        raise HTTPException(404, "File not found")
    try:
        return audio_io.file_peaks(p, body.buckets)
    except Exception as exc:
        raise HTTPException(400, f"Can't read audio: {exc}")


@app.post("/audio/info")
def audio_info(body: AudioIn):
    p = Path(body.path).expanduser()
    if not p.exists():
        raise HTTPException(404, "File not found")
    info = audio_io.probe(p)
    return {"duration": info.duration, "sample_rate": info.sample_rate,
            "channels": info.channels, "codec": info.codec}


@app.get("/models")
def list_models():
    return models.list()


@app.post("/models/{model_id}/download")
def download_model(model_id: str):
    if model_id not in CATALOG:
        raise HTTPException(404, "Unknown model")
    models.download_async(model_id)
    return models.status(model_id)


@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    if model_id not in CATALOG:
        raise HTTPException(404, "Unknown model")
    models.delete(model_id)
    return models.status(model_id)


# ---------------------------------------------------------------- launch ---

def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Vaani engine server")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    args = parser.parse_args(argv)

    ensure_dirs()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", args.port))
    sock.listen(64)
    port = sock.getsockname()[1]
    get_manager()
    # The Mac app waits for this line to learn the port.
    print(f"READY port={port}", flush=True)
    sys.stdout.flush()
    config = uvicorn.Config(app, log_level="warning", access_log=False)
    uvicorn.Server(config).run(sockets=[sock])


if __name__ == "__main__":
    main()
