import time
from pathlib import Path

import numpy as np
import pytest

from fakes import make_registry
from vaani_engine import audio_io
from vaani_engine.jobs import JobManager, JobStore
from vaani_engine.providers.base import Stage


def asr_pipeline(ctx):
    transcript = ctx.run_stage(Stage.ASR, lambda p: p.transcribe(np.zeros(16000), ctx.src_lang))
    return {"text": transcript.text}


def wait(job, timeout=10):
    deadline = time.time() + timeout
    while job.status in ("queued", "running") and time.time() < deadline:
        time.sleep(0.02)
    return job


@pytest.fixture
def wav(tmp_path) -> Path:
    return audio_io.save(tmp_path / "in.wav", np.zeros((1, 16000), dtype=np.float32), 16000)


def manager(tmp_path, **kw):
    return JobManager(make_registry(**kw), {"transcribe": asr_pipeline}, JobStore(tmp_path / "jobs.db"))


def test_auto_mode_falls_back_and_records_provider(tmp_path, wav):
    mgr = manager(tmp_path, keys={"sarvam": "k"}, failing_cloud_asr=True)
    job = wait(mgr.submit("transcribe", str(wav), {"mode": "auto", "src_lang": "mr"}))
    assert job.status == "done"
    assert job.providers_used["asr"] == "local_asr"
    assert any("Falling back" in line for line in job.log)


def test_explicit_cloud_mode_does_not_fall_back_silently(tmp_path, wav):
    mgr = manager(tmp_path, keys={"sarvam": "k"}, failing_cloud_asr=True)
    job = wait(mgr.submit("transcribe", str(wav), {"mode": "cloud", "src_lang": "mr"}))
    assert job.status == "failed"
    assert job.error["stage"] == "asr"
    assert job.error["local_alternative"] == "local_asr"


def test_blocked_plan_is_rejected_before_running(tmp_path, wav):
    mgr = manager(tmp_path)
    with pytest.raises(PermissionError):
        mgr.submit("transcribe", str(wav), {"mode": "cloud", "src_lang": "mr"})


def test_jobs_persist(tmp_path, wav):
    mgr = manager(tmp_path)
    job = wait(mgr.submit("transcribe", str(wav), {"mode": "local", "src_lang": "hi"}))
    assert job.status == "done" and job.outputs == {"text": "namaste"}
    reloaded = JobStore(tmp_path / "jobs.db").load_all()
    assert reloaded[0].id == job.id and reloaded[0].status == "done"


def test_untranslatable_lines_are_retried_and_reported(tmp_path, wav):
    from vaani_engine.jobs import Job, JobContext
    from vaani_engine.pipelines.transcribe import translate
    from vaani_engine.providers.base import Segment, Transcript

    mgr = manager(tmp_path, keys={"anthropic": "k"})
    calls = []

    def transliterating(texts, src, tgt, durations=None, progress=None):
        """Like the small local model on a garbled transcript: spells it out instead of translating."""
        calls.append(len(texts))
        return ["Tar tumhie aisha growth karnu shapdana main je thod karnaa" if "bad" in t
                else "You can do it well." for t in texts]

    mgr.registry.get("llm_mt").translate = transliterating
    plan = mgr.registry.resolve_plan("transcribe", "mr", "en", "local", overrides={"translate": "llm_mt"})
    job = Job(id="t1", type="transcribe", input_path=str(wav), params={}, plan=plan.to_dict(), out_dir=str(tmp_path))
    ctx = JobContext(mgr, job, plan)
    transcript = Transcript("mr", [Segment(0, 1, "good line"), Segment(1, 2, "bad line")])

    assert translate(ctx, transcript, "en", 0, 1) == [1]
    assert calls == [2, 1]                       # one batch, then one retry for the bad line
    assert transcript.segments[0].translation == "You can do it well."
