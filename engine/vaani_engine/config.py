"""Paths and in-memory settings shared by the engine.

API keys are never written to disk by the engine. The Mac app keeps them in
the Keychain and hands them over through environment variables at launch or
through POST /settings.
"""

from __future__ import annotations

import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "VaaniStudio"

ENGINE_ROOT = Path(__file__).resolve().parents[1]
RUNTIMES_DIR = ENGINE_ROOT / "runtimes"

SUPPORT_DIR = Path(
    os.environ.get("VAANI_HOME", Path.home() / "Library" / "Application Support" / APP_NAME)
)
MODELS_DIR = SUPPORT_DIR / "models"
JOBS_DB = SUPPORT_DIR / "jobs.db"
OUTPUT_DIR = Path(os.environ.get("VAANI_OUTPUT", Path.home() / "Music" / APP_NAME))

# Names of the keys the engine understands, mapped to their env variables.
KEY_ENV = {
    "sarvam": "VAANI_SARVAM_API_KEY",
    "openai": "VAANI_OPENAI_API_KEY",
    "anthropic": "VAANI_ANTHROPIC_API_KEY",
    "huggingface": "VAANI_HF_TOKEN",
}


@dataclass
class Settings:
    keys: dict[str, str] = field(default_factory=dict)
    offline: bool = False
    output_dir: Path = OUTPUT_DIR
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_env(cls) -> "Settings":
        keys = {name: os.environ[env] for name, env in KEY_ENV.items() if os.environ.get(env)}
        offline = os.environ.get("VAANI_OFFLINE", "0") == "1"
        return cls(keys=keys, offline=offline)

    def has_key(self, name: str) -> bool:
        return bool(self.keys.get(name))

    def key(self, name: str) -> str | None:
        return self.keys.get(name) or None

    def update(self, keys: dict[str, str | None] | None = None, offline: bool | None = None,
               output_dir: str | None = None) -> None:
        with self._lock:
            for name, value in (keys or {}).items():
                if name not in KEY_ENV:
                    continue
                if value:
                    self.keys[name] = value
                else:
                    self.keys.pop(name, None)
            if offline is not None:
                self.offline = offline
            if output_dir:
                self.output_dir = Path(output_dir).expanduser()
        hf = self.keys.get("huggingface")
        if hf:
            os.environ["HF_TOKEN"] = hf
        else:
            os.environ.pop("HF_TOKEN", None)

    def key_status(self) -> dict[str, bool]:
        return {name: self.has_key(name) for name in KEY_ENV}


settings = Settings.from_env()


def find_tool(name: str) -> str:
    """Locate a command-line tool. Apps launched from Finder don't get the shell PATH."""
    override = os.environ.get(f"VAANI_{name.upper()}")
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
        candidate = Path(prefix) / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"{name} not found. Install it with: brew install {name}")


def ensure_dirs() -> None:
    for d in (SUPPORT_DIR, MODELS_DIR, settings.output_dir):
        d.mkdir(parents=True, exist_ok=True)
