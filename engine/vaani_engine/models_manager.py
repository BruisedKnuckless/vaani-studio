"""Catalog of downloadable models, with download, delete and status.

Models with a Hugging Face `repo` are stored in MODELS_DIR/<id>. Models with
repo=None are fetched by their library on first use (small, auto-managed).
"""

from __future__ import annotations

import shutil
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import MODELS_DIR, settings


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    purpose: str
    repo: str | None
    size_mb: int
    license: str
    gated: bool = False
    runtime: str = "core"
    allow_patterns: tuple[str, ...] | None = None
    commercial_ok: bool = True
    # Post-download step, e.g. "whisper_hf" converts a transformers Whisper checkpoint to MLX.
    convert: str | None = None


CATALOG: dict[str, ModelSpec] = {m.id: m for m in [
    ModelSpec("deepfilternet3", "DeepFilterNet 3", "Fast noise removal", None, 10, "MIT/Apache-2.0"),
    ModelSpec("mossformer2-se", "MossFormer2 SE 48K (MLX)", "High-quality noise removal",
              "starkdmi/MossFormer2-SE", 221, "Apache-2.0"),
    ModelSpec("htdemucs", "Demucs (htdemucs)", "Separate voice from music/background", None, 85, "MIT"),
    ModelSpec("whisper-large-v3-turbo", "Whisper large-v3 turbo (MLX)", "Transcribe English, Hindi, auto-detect",
              "mlx-community/whisper-large-v3-turbo", 1614, "MIT"),
    ModelSpec("whisper-marathi", "Whisper large-v2 Marathi (MLX)", "Transcribe Marathi accurately on this Mac",
              "DrishtiSharma/whisper-large-v2-marathi", 6174, "Apache-2.0",
              allow_patterns=("*.json", "*.txt", "pytorch_model.bin"), convert="whisper_hf"),
    ModelSpec("indic-conformer", "IndicConformer 600M", "Transcribe Marathi and Hindi",
              "ai4bharat/indic-conformer-600m-multilingual", 2557, "MIT", gated=True),
    ModelSpec("indictrans2-en-indic", "IndicTrans2 En→Indic (200M)", "Translate English → Hindi/Marathi",
              "ai4bharat/indictrans2-en-indic-dist-200M", 1110, "MIT", gated=True,
              allow_patterns=("*.json", "*.py", "model.safetensors", "*.SRC", "*.TGT", "*.model")),
    ModelSpec("indictrans2-indic-en", "IndicTrans2 Indic→En (200M)", "Translate Hindi/Marathi → English",
              "ai4bharat/indictrans2-indic-en-dist-200M", 925, "MIT", gated=True,
              allow_patterns=("*.json", "*.py", "model.safetensors", "*.SRC", "*.TGT", "*.model")),
    ModelSpec("local-llm-large", "Gemma 3 12B (MLX 4-bit)", "On-device translation: English, Hindi, Marathi",
              "mlx-community/gemma-3-12b-it-4bit", 8068, "Gemma terms of use"),
    ModelSpec("local-llm", "Qwen3 4B Instruct (MLX 4-bit)", "Lightweight on-device translation and Hinglish",
              "mlx-community/Qwen3-4B-Instruct-2507-4bit", 2279, "Apache-2.0"),
    ModelSpec("chatterbox-mlx", "Chatterbox multilingual (MLX)", "Voice-cloned speech in English and Hindi",
              "mlx-community/chatterbox-multilingual-v3", 2711, "MIT", runtime="mlxvoice"),
    ModelSpec("chatterbox-vc", "Chatterbox voice conversion", "Makes generated speech sound like the speaker",
              "ResembleAI/chatterbox", 1057, "MIT", runtime="voice", allow_patterns=("s3gen.pt", "conds.pt")),
    ModelSpec("mms-tts-mar", "MMS-TTS Marathi", "Marathi speech (then matched to the speaker's voice)",
              "facebook/mms-tts-mar", 146, "CC-BY-NC-4.0 (non-commercial)", runtime="voice", commercial_ok=False,
              allow_patterns=("*.json", "model.safetensors")),
]}


class ModelsManager:
    def __init__(self, root: Path = MODELS_DIR):
        self.root = root
        self._downloads: dict[str, dict] = {}
        self._lock = threading.Lock()

    def path(self, model_id: str) -> Path:
        return self.root / model_id

    def is_downloaded(self, model_id: str) -> bool:
        spec = CATALOG[model_id]
        if spec.repo is None:
            return True
        return (self.path(model_id) / ".complete").exists()

    def local_path(self, model_id: str) -> Path | None:
        return self.path(model_id) if self.is_downloaded(model_id) else None

    def disk_mb(self, model_id: str) -> int:
        p = self.path(model_id)
        if not p.exists():
            return 0
        return int(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6)

    def status(self, model_id: str) -> dict:
        spec = CATALOG[model_id]
        dl = self._downloads.get(model_id)
        state = "managed" if spec.repo is None else ("downloaded" if self.is_downloaded(model_id) else "missing")
        progress = None
        if dl and dl["state"] == "downloading":
            state = "downloading"
            progress = min(0.99, self.disk_mb(model_id) / max(spec.size_mb, 1))
        elif dl and dl["state"] == "error" and state == "missing":
            state = "error"
        return asdict(spec) | {
            "state": state, "progress": progress, "disk_mb": self.disk_mb(model_id),
            "error": dl.get("error") if dl else None,
            "terms_url": f"https://huggingface.co/{spec.repo}" if spec.gated else None,
        }

    def list(self) -> list[dict]:
        return [self.status(mid) for mid in CATALOG]

    def download(self, model_id: str) -> Path:
        """Blocking download. Safe to call from several threads; only one runs per model."""
        spec = CATALOG[model_id]
        if spec.repo is None or self.is_downloaded(model_id):
            return self.path(model_id)
        with self._lock:
            if self._downloads.get(model_id, {}).get("state") == "downloading":
                event = self._downloads[model_id]["event"]
            else:
                event = None
                self._downloads[model_id] = {"state": "downloading", "event": threading.Event()}
        if event is not None:
            event.wait()
            if not self.is_downloaded(model_id):
                raise RuntimeError(self._downloads[model_id].get("error") or f"Download of {spec.name} failed")
            return self.path(model_id)

        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        record = self._downloads[model_id]
        try:
            snapshot_download(
                repo_id=spec.repo, local_dir=self.path(model_id),
                allow_patterns=list(spec.allow_patterns) if spec.allow_patterns else None,
                token=settings.key("huggingface"),
            )
            if spec.convert == "whisper_hf":
                from .convert_whisper import convert

                convert(self.path(model_id), remove_source=True)  # ~6 GB fp32 → ~3 GB MLX fp16
            (self.path(model_id) / ".complete").write_text(spec.repo)
            record["state"] = "done"
        except (GatedRepoError, RepositoryNotFoundError) as exc:
            record.update(state="error", error=(
                f"{spec.name} needs you to accept its terms on Hugging Face "
                f"(https://huggingface.co/{spec.repo}) and add a Hugging Face token in Settings."
            ))
            raise RuntimeError(record["error"]) from exc
        except Exception as exc:
            record.update(state="error", error=f"Download of {spec.name} failed: {exc}")
            raise RuntimeError(record["error"]) from exc
        finally:
            record["event"].set()
        return self.path(model_id)

    def download_async(self, model_id: str) -> None:
        threading.Thread(target=self._safe_download, args=(model_id,), daemon=True).start()

    def _safe_download(self, model_id: str) -> None:
        try:
            self.download(model_id)
        except Exception:
            pass  # the error is kept in self._downloads and shown by status()

    def delete(self, model_id: str) -> None:
        if CATALOG[model_id].repo is None:
            return
        shutil.rmtree(self.path(model_id), ignore_errors=True)
        self._downloads.pop(model_id, None)


models = ModelsManager()


def readiness_for(model_ids: tuple[str, ...]) -> tuple[str, str]:
    """Shared readiness check for local providers."""
    missing = [CATALOG[m] for m in model_ids if not models.is_downloaded(m)]
    if not missing:
        return "ready", ""
    gated = [m for m in missing if m.gated]
    if gated and not settings.has_key("huggingface"):
        names = ", ".join(m.name for m in gated)
        return "unavailable", (f"{names} needs a free Hugging Face token: accept the model's terms on "
                               "huggingface.co, then add the token in Settings")
    total = sum(m.size_mb for m in missing)
    size = f"{total / 1000:.1f} GB" if total >= 1000 else f"{total} MB"
    return "needs_download", f"Downloads {size} on first use"
