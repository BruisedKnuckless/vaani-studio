"""Runs model families that need their own Python environment as child processes.

Each runtime lives in engine/runtimes/<name>/ with its own .venv and a worker.py
speaking JSON lines over stdin/stdout. A worker starts on first use and exits
after sitting idle, so its models don't hold memory between jobs.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from ..config import RUNTIMES_DIR


class RuntimeError_(RuntimeError):
    pass


class WorkerHost:
    _hosts: dict[str, "WorkerHost"] = {}

    @classmethod
    def get(cls, runtime: str) -> "WorkerHost":
        if runtime not in cls._hosts:
            cls._hosts[runtime] = WorkerHost(runtime)
        return cls._hosts[runtime]

    def __init__(self, runtime: str, idle_timeout: float = 90.0):
        self.runtime = runtime
        self.dir = RUNTIMES_DIR / runtime
        self.python = self.dir / ".venv" / "bin" / "python"
        self.idle_timeout = idle_timeout
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=60)
        self._lock = threading.Lock()
        self._next_id = 0
        self._last_used = 0.0
        self._idle_timer: threading.Timer | None = None

    @property
    def installed(self) -> bool:
        return self.python.exists() and (self.dir / "worker.py").exists()

    def _start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        if not self.installed:
            raise RuntimeError_(f"The '{self.runtime}' voice runtime isn't installed. Run scripts/bootstrap.sh.")
        env = os.environ | {"PYTHONUNBUFFERED": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                            "TOKENIZERS_PARALLELISM": "false"}
        self._proc = subprocess.Popen(
            [str(self.python), str(self.dir / "worker.py")], cwd=self.dir, env=env, text=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1,
        )
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self._proc.stdout, self._lines), daemon=True).start()
        threading.Thread(target=self._drain, args=(self._proc.stderr,), daemon=True).start()

    @staticmethod
    def _pump(stream, sink: queue.Queue) -> None:
        for line in stream:
            sink.put(line)
        sink.put(None)

    def _drain(self, stream) -> None:
        for line in stream:
            self._stderr.append(line.rstrip())

    def call(self, method: str, params: dict | None = None, timeout: float = 900.0) -> dict:
        with self._lock:
            self._start()
            self._next_id += 1
            request_id = self._next_id
            assert self._proc and self._proc.stdin
            self._proc.stdin.write(json.dumps({"id": request_id, "method": method, "params": params or {}}) + "\n")
            self._proc.stdin.flush()
            deadline = time.time() + timeout
            while True:
                try:
                    line = self._lines.get(timeout=max(0.1, deadline - time.time()))
                except queue.Empty:
                    self.stop()
                    raise RuntimeError_(f"The {self.runtime} runtime timed out on {method}")
                if line is None:
                    tail = "\n".join(list(self._stderr)[-6:])
                    self._proc = None
                    raise RuntimeError_(f"The {self.runtime} runtime stopped unexpectedly.\n{tail}")
                try:
                    reply = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if reply.get("id") != request_id:
                    continue
                self._last_used = time.time()
                self._schedule_idle_stop()
                if "error" in reply:
                    raise RuntimeError_(reply["error"])
                return reply.get("result") or {}

    def _schedule_idle_stop(self) -> None:
        if self._idle_timer:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(self.idle_timeout, self._stop_if_idle)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _stop_if_idle(self) -> None:
        if time.time() - self._last_used >= self.idle_timeout - 1 and self._lock.acquire(blocking=False):
            try:
                self.stop()
            finally:
                self._lock.release()

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            try:
                proc.stdin.close()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    @classmethod
    def stop_all(cls) -> None:
        for host in cls._hosts.values():
            host.stop()


def runtime_dir(name: str) -> Path:
    return RUNTIMES_DIR / name
