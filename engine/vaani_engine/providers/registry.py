"""Chooses which provider runs each step of a job.

The user picks a mode per upload: "local" (on this Mac), "cloud" (APIs) or
"auto". `resolve_plan` turns that choice, plus optional per-step overrides,
into one concrete provider per step and reports anything that blocks the job
(a missing key, an unavailable runtime) so the app can show it before Start.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from ..config import Settings
from .base import STAGE_LABELS, Kind, Provider, Stage

Mode = Literal["local", "cloud", "auto"]
JobType = Literal["enhance", "transcribe", "dub"]

POLISH_ID = "dsp_polish"
SERVICE_NAMES = {"sarvam": "Sarvam AI", "anthropic": "Anthropic", "openai": "OpenAI", "huggingface": "Hugging Face"}

# Stages with no provider choice: they always run on this Mac.
FIXED_STAGES = {Stage.POLISH: (POLISH_ID, "Built-in EQ, compressor & loudness")}

# Enhance's "quality" option picks the noise remover unless the user overrides it.
QUALITY_DENOISERS = {"fast": "deepfilternet", "high": "mossformer2", "basic": "basic_dsp"}


@dataclass
class Choice:
    id: str
    name: str
    kind: Kind
    status: str


@dataclass
class StagePlan:
    stage: str
    label: str
    provider_id: str
    provider_name: str
    kind: Kind
    # ready | needs_download | missing_key | unavailable | local_only
    status: str
    note: str = ""
    fallback_id: str | None = None
    local_alternative: str | None = None
    alternatives: list[Choice] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        return self.status in ("missing_key", "unavailable")


@dataclass
class Plan:
    job_type: str
    mode: str
    src_lang: str
    tgt_lang: str | None
    steps: list[StagePlan]
    offline: bool

    @property
    def uploads(self) -> bool:
        return any(s.kind == "cloud" for s in self.steps)

    @property
    def blocking(self) -> bool:
        return any(s.blocking for s in self.steps)

    def step(self, stage: Stage) -> StagePlan | None:
        return next((s for s in self.steps if s.stage == stage.value), None)

    def to_dict(self) -> dict:
        return {
            "job_type": self.job_type, "mode": self.mode, "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang, "offline": self.offline, "uploads": self.uploads,
            "blocking": self.blocking, "steps": [asdict(s) | {"blocking": s.blocking} for s in self.steps],
            "summary": self.summary(),
        }

    def summary(self) -> str:
        parts = []
        for s in self.steps:
            where = "cloud" if s.kind == "cloud" else "this Mac"
            parts.append(f"{s.provider_name} ({where})")
        return " → ".join(parts)


def stages_for(job_type: str, src: str, tgt: str | None, options: dict) -> list[Stage]:
    if job_type == "enhance":
        stages = [Stage.SEPARATE] if options.get("isolate_vocals") else []
        return stages + [Stage.DENOISE, Stage.POLISH]
    if job_type == "transcribe":
        stages = [] if options.get("segments") else [Stage.ASR]
        if tgt and tgt != src:
            stages.append(Stage.TRANSLATE)
        return stages
    if job_type == "dub":
        stages = [Stage.SEPARATE, Stage.ASR, Stage.TRANSLATE, Stage.TTS]
        if options.get("voice_match"):
            stages.append(Stage.VOICE_MATCH)
        return stages
    raise ValueError(f"Unknown job type {job_type}")


def auto_preference(stage: Stage, src: str, tgt: str | None) -> list[Kind]:
    """Which side Auto mode tries first for each step."""
    if stage == Stage.ASR:
        # Sarvam is clearly better for Marathi and code-mixed Hinglish.
        return ["cloud", "local"] if src in ("mr", "hinglish") else ["local", "cloud"]
    if stage == Stage.TRANSLATE:
        # LLMs handle context, Hinglish and dubbing length limits best.
        return ["cloud", "local"]
    if stage == Stage.TTS:
        return ["cloud", "local"] if tgt == "mr" else ["local", "cloud"]
    return ["local", "cloud"]


class Registry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._providers: dict[str, Provider] = {}

    def register(self, provider: Provider) -> Provider:
        self._providers[provider.id] = provider
        return provider

    def get(self, provider_id: str) -> Provider:
        if provider_id not in self._providers:
            raise KeyError(f"Unknown provider {provider_id}")
        return self._providers[provider_id]

    def all(self) -> list[Provider]:
        return list(self._providers.values())

    def for_stage(self, stage: Stage, src: str, tgt: str | None, kind: Kind | None = None) -> list[Provider]:
        found = [p for p in self._providers.values()
                 if p.stage == stage and p.supports(src, tgt) and (kind is None or p.kind == kind)]
        return sorted(found, key=lambda p: p.rank_for(src, tgt))

    def status_of(self, provider: Provider, src: str | None = None, tgt: str | None = None) -> tuple[str, str]:
        if provider.kind == "cloud":
            if self.settings.offline:
                return "unavailable", "Offline mode is on"
            if provider.key_name and not self.settings.has_key(provider.key_name):
                service = SERVICE_NAMES.get(provider.key_name, provider.key_name)
                return "missing_key", f"Add your {service} API key in Settings"
        return provider.readiness(src, tgt)

    def usable(self, provider: Provider, src: str | None = None, tgt: str | None = None) -> bool:
        return self.status_of(provider, src, tgt)[0] in ("ready", "needs_download")

    def _best(self, stage: Stage, src: str, tgt: str | None, kind: Kind) -> Provider | None:
        candidates = self.for_stage(stage, src, tgt, kind)
        return next((p for p in candidates if self.usable(p, src, tgt)), None)

    def _step(self, stage: Stage, provider: Provider, status: str | None = None, note: str | None = None,
              src: str = "auto", tgt: str | None = None) -> StagePlan:
        state, why = self.status_of(provider, src, tgt)
        alternatives = [Choice(p.id, p.name, p.kind, self.status_of(p, src, tgt)[0])
                        for p in self.for_stage(stage, src, tgt)]
        local = self._best(stage, src, tgt, "local")
        return StagePlan(
            stage=stage.value, label=STAGE_LABELS[stage], provider_id=provider.id,
            provider_name=provider.name, kind=provider.kind, status=status or state,
            note=note if note is not None else why, alternatives=alternatives,
            local_alternative=local.id if local and provider.kind == "cloud" else None,
        )

    def resolve_plan(self, job_type: str, src: str = "auto", tgt: str | None = None,
                     mode: Mode = "auto", overrides: dict[str, str] | None = None,
                     options: dict | None = None) -> Plan:
        options = options or {}
        overrides = overrides or {}
        offline = self.settings.offline
        effective_mode: Mode = "local" if offline else mode
        steps: list[StagePlan] = []

        for stage in stages_for(job_type, src, tgt, options):
            if stage in FIXED_STAGES:
                pid, name = FIXED_STAGES[stage]
                steps.append(StagePlan(stage.value, STAGE_LABELS[stage], pid, name, "local", "ready"))
                continue

            override = overrides.get(stage.value)
            if not override and stage == Stage.DENOISE:
                override = QUALITY_DENOISERS.get(options.get("quality", ""))
            if override and override in self._providers:
                provider = self._providers[override]
                if offline and provider.kind == "cloud":
                    provider = None
                elif provider.stage == stage:
                    steps.append(self._step(stage, provider, src=src, tgt=tgt))
                    continue

            local = self._best(stage, src, tgt, "local")
            cloud = self._best(stage, src, tgt, "cloud")
            any_cloud = self.for_stage(stage, src, tgt, "cloud")
            any_local = self.for_stage(stage, src, tgt, "local")

            if effective_mode == "local":
                if local:
                    steps.append(self._step(stage, local, src=src, tgt=tgt))
                elif any_local:
                    steps.append(self._step(stage, any_local[0], src=src, tgt=tgt))
                else:
                    steps.append(self._unavailable(stage, "No on-device option for this language yet"))
                continue

            if effective_mode == "cloud":
                if cloud:
                    steps.append(self._step(stage, cloud, src=src, tgt=tgt))
                elif any_cloud:
                    # A cloud service exists but can't be used (usually a missing key).
                    steps.append(self._step(stage, any_cloud[0], src=src, tgt=tgt))
                elif local:
                    steps.append(self._step(stage, local, status="local_only",
                                            note="No cloud service for this step, so it runs on this Mac",
                                            src=src, tgt=tgt))
                else:
                    steps.append(self._unavailable(stage, "No provider supports this language"))
                continue

            # Auto: preferred side first, the other side as fallback.
            order = auto_preference(stage, src, tgt)
            picks = {"local": local, "cloud": cloud}
            chosen = next((picks[k] for k in order if picks[k]), None)
            if not chosen:
                steps.append(self._unavailable(stage, "No usable provider (add an API key or download a model)"))
                continue
            step = self._step(stage, chosen, src=src, tgt=tgt)
            other = cloud if chosen.kind == "local" else local
            step.fallback_id = other.id if other else None
            steps.append(step)

        return Plan(job_type, mode, src, tgt, steps, offline)

    def _unavailable(self, stage: Stage, note: str) -> StagePlan:
        return StagePlan(stage.value, STAGE_LABELS[stage], "", "Not available", "local", "unavailable", note)

    def cloud_available_for(self, job_type: str) -> bool:
        """False when no step of this job type has any cloud provider (e.g. Enhance)."""
        stages = stages_for(job_type, "auto", "en", {"isolate_vocals": True, "voice_match": True})
        return any(p.kind == "cloud" for p in self._providers.values() if p.stage in stages)
