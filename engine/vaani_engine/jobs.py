"""Job queue, persistence (SQLite) and progress events.

Jobs run one at a time on a background thread so that only one heavy model
needs to be in memory on a 16 GB Mac.
"""

from __future__ import annotations

import asyncio
import json
import queue
import sqlite3
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import JOBS_DB, settings
from .providers.base import ProviderError, Stage
from .providers.registry import Plan, Registry, StagePlan


class Cancelled(Exception):
    pass


class StageFailed(Exception):
    def __init__(self, step: StagePlan, cause: Exception):
        super().__init__(f"{step.label} failed with {step.provider_name}: {cause}")
        self.step = step
        self.cause = cause


@dataclass
class Job:
    id: str
    type: str
    input_path: str
    params: dict
    plan: dict
    out_dir: str
    status: str = "queued"  # queued | running | done | failed | cancelled
    progress: float = 0.0
    message: str = "Waiting"
    providers_used: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    error: dict | None = None
    log: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class JobStore:
    def __init__(self, path: Path = JOBS_DB):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created_at REAL, data TEXT)")
        self._conn.commit()

    def save(self, job: Job) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO jobs VALUES (?, ?, ?)",
                               (job.id, job.created_at, json.dumps(job.to_dict(), default=str)))
            self._conn.commit()

    def load_all(self, limit: int = 200) -> list[Job]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [Job(**json.loads(r[0])) for r in rows]

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()


class JobContext:
    """What a pipeline sees while it runs: options, progress, and provider calls
    that honour the per-upload Local/Cloud/Auto choice."""

    def __init__(self, manager: "JobManager", job: Job, plan: Plan):
        self._manager = manager
        self.registry = manager.registry
        self.job = job
        self.plan = plan
        self.options: dict = job.params.get("options", {})
        self.src_lang: str = job.params.get("src_lang", "auto")
        self.tgt_lang: str | None = job.params.get("tgt_lang")
        self.out_dir = Path(job.out_dir)
        self.input_path = Path(job.input_path)
        self._cancel = threading.Event()
        self._announced: set[Stage] = set()

    # progress -----------------------------------------------------------
    def progress(self, fraction: float, message: str | None = None) -> None:
        if self._cancel.is_set():
            raise Cancelled()
        self.job.progress = max(self.job.progress, min(1.0, fraction))
        if message:
            self.job.message = message
        self._manager._touch(self.job)

    def span(self, start: float, end: float, label: str) -> Callable[[float, str], None]:
        """Progress function for a sub-step mapped into [start, end] of the job."""
        def report(fraction: float, message: str = "") -> None:
            self.progress(start + (end - start) * max(0.0, min(1.0, fraction)),
                          f"{label}: {message}" if message else label)
        return report

    def log(self, line: str) -> None:
        self.job.log.append(f"{datetime.now():%H:%M:%S} {line}")
        self._manager._touch(self.job, persist=True)

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    # providers ----------------------------------------------------------
    def has_stage(self, stage: Stage) -> bool:
        return self.plan.step(stage) is not None

    def provider_for(self, stage: Stage):
        step = self.plan.step(stage)
        if step is None or not step.provider_id:
            raise ProviderError(f"No provider planned for {stage.value}")
        return self._manager.registry.get(step.provider_id)

    def release(self, stage: Stage) -> None:
        """Free a heavy model once its step is finished (keeps memory low on long jobs)."""
        provider_id = self.job.providers_used.get(stage.value)
        if provider_id:
            provider = self.registry.get(provider_id)
            if provider.heavy:
                provider.unload()

    def run_stage(self, stage: Stage, fn: Callable[[Any], Any]) -> Any:
        """Call `fn(provider)` with the planned provider. In Auto mode a failure
        falls back to the other side; in Local/Cloud mode it is reported so the
        user can choose (the app offers "Run this step locally")."""
        step = self.plan.step(stage)
        if step is None:
            raise ProviderError(f"Step {stage.value} is not part of this job")
        provider = self.provider_for(stage)
        if stage not in self._announced:  # steps called per line (TTS) log once
            self._announced.add(stage)
            self.log(f"{step.label}: {provider.name} ({'cloud' if provider.kind == 'cloud' else 'this Mac'})")
        try:
            result = fn(provider)
            self.job.providers_used[stage.value] = provider.id
            return result
        except Cancelled:
            raise
        except Exception as exc:
            self.log(f"{step.label} failed with {provider.name}: {exc}")
            if self.plan.mode == "auto" and step.fallback_id and not self.plan.offline:
                fallback = self._manager.registry.get(step.fallback_id)
                self.log(f"Falling back to {fallback.name}")
                result = fn(fallback)
                self.job.providers_used[stage.value] = fallback.id
                return result
            raise StageFailed(step, exc) from exc


class JobManager:
    def __init__(self, registry: Registry, pipelines: dict[str, Callable[[JobContext], dict]],
                 store: JobStore | None = None):
        self.registry = registry
        self.pipelines = pipelines
        self.store = store or JobStore()
        self.jobs: dict[str, Job] = {}
        self._contexts: dict[str, JobContext] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._sub_lock = threading.Lock()
        self._last_persist: dict[str, float] = {}
        for job in self.store.load_all():
            if job.status in ("queued", "running"):
                job.status, job.message = "failed", "Interrupted when the engine stopped"
                self.store.save(job)
            self.jobs[job.id] = job
        threading.Thread(target=self._worker, daemon=True, name="vaani-jobs").start()

    # submission ----------------------------------------------------------
    def submit(self, job_type: str, input_path: str, params: dict) -> Job:
        if job_type not in self.pipelines:
            raise ValueError(f"Unknown job type {job_type}")
        path = Path(input_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        plan = self.registry.resolve_plan(
            job_type, params.get("src_lang", "auto"), params.get("tgt_lang"),
            params.get("mode", "auto"), params.get("overrides"), params.get("options"),
        )
        if plan.blocking:
            blocked = [f"{s.label}: {s.note}" for s in plan.steps if s.blocking]
            raise PermissionError("Can't start yet. " + " · ".join(blocked))
        job_id = uuid.uuid4().hex[:10]
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = settings.output_dir / f"{stamp}_{_safe(path.stem)}_{job_type}_{job_id[:4]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, type=job_type, input_path=str(path), params=params,
                  plan=plan.to_dict(), out_dir=str(out_dir))
        self.jobs[job_id] = job
        self._contexts[job_id] = JobContext(self, job, plan)
        self.store.save(job)
        self._emit(job)
        self._queue.put(job_id)
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        ctx = self._contexts.get(job_id)
        if ctx:
            ctx._cancel.set()
        if job.status == "queued":
            job.status, job.message = "cancelled", "Cancelled"
            self._touch(job, persist=True)
        return job

    def delete(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job and job.status in ("queued", "running"):
            self.cancel(job_id)
        self.jobs.pop(job_id, None)
        self.store.delete(job_id)

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    # running -------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.jobs.get(job_id)
            ctx = self._contexts.get(job_id)
            if not job or not ctx or job.status != "queued":
                continue
            self.run_now(ctx)

    def run_now(self, ctx: JobContext) -> Job:
        job = ctx.job
        job.status, job.message = "running", "Starting"
        self._touch(job, persist=True)
        try:
            job.outputs = self.pipelines[job.type](ctx) or {}
            job.status, job.progress, job.message = "done", 1.0, "Done"
        except Cancelled:
            job.status, job.message = "cancelled", "Cancelled"
        except StageFailed as exc:
            job.status, job.message = "failed", str(exc)
            job.error = {"message": str(exc), "stage": exc.step.stage, "provider": exc.step.provider_id,
                         "kind": exc.step.kind, "local_alternative": exc.step.local_alternative}
            ctx.log(traceback.format_exc(limit=3))
        except Exception as exc:
            job.status, job.message = "failed", f"{type(exc).__name__}: {exc}"
            job.error = {"message": job.message}
            ctx.log(traceback.format_exc(limit=5))
        finally:
            self._contexts.pop(job.id, None)
            self._release_models(job)
            self._touch(job, persist=True)
        return job

    def _release_models(self, job: Job) -> None:
        for provider_id in set(job.providers_used.values()):
            try:
                provider = self.registry.get(provider_id)
                if provider.heavy:
                    provider.unload()
            except Exception:
                pass

    # events --------------------------------------------------------------
    def _touch(self, job: Job, persist: bool = False) -> None:
        job.updated_at = time.time()
        now = job.updated_at
        if persist or now - self._last_persist.get(job.id, 0) > 2.0:
            self.store.save(job)
            self._last_persist[job.id] = now
        self._emit(job)

    def _emit(self, job: Job) -> None:
        payload = job.to_dict()
        with self._sub_lock:
            subscribers = list(self._subscribers)
        for loop, q in subscribers:
            try:
                loop.call_soon_threadsafe(_offer, q, payload)
            except RuntimeError:
                pass  # loop closed

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        with self._sub_lock:
            self._subscribers.append((asyncio.get_running_loop(), q))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._sub_lock:
            self._subscribers = [(l, s) for l, s in self._subscribers if s is not q]


def _offer(q: asyncio.Queue, payload: dict) -> None:
    try:
        q.put_nowait(payload)
    except asyncio.QueueFull:
        pass  # a slow listener only misses intermediate progress


def _safe(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return cleaned[:40] or "audio"
