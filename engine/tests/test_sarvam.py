"""Sarvam-105B translator retry logic, with the HTTP layer faked."""

import json

import pytest

from vaani_engine.providers.base import ProviderError
from vaani_engine.providers.cloud import sarvam


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def reply(finish, translations=None):
    content = json.dumps({"translations": translations}) if translations is not None else ""
    return FakeResponse({"choices": [{"finish_reason": finish, "message": {"content": content}}],
                         "usage": {"completion_tokens": 100, "completion_tokens_details": {"reasoning_tokens": 90}}})


def test_runs_out_while_reasoning_then_retries_without_reasoning(monkeypatch):
    calls = []

    def fake(method, url, **kw):
        calls.append(kw["json"]["reasoning_effort"])
        n = len(json.loads(kw["json"]["messages"][1]["content"])["lines"])
        return reply("length") if kw["json"]["reasoning_effort"] == "low" else reply("stop", ["ok"] * n)

    monkeypatch.setattr(sarvam, "request_with_retry", fake)
    out = sarvam.SarvamLLMTranslator().translate(["a", "b", "c"], "mr", "en")
    assert out == ["ok", "ok", "ok"]
    assert calls == ["low", None]


def test_splits_batch_when_both_attempts_run_out(monkeypatch):
    sizes = []

    def fake(method, url, **kw):
        n = len(json.loads(kw["json"]["messages"][1]["content"])["lines"])
        sizes.append(n)
        return reply("length") if n > 1 else reply("stop", ["x"])

    monkeypatch.setattr(sarvam, "request_with_retry", fake)
    assert sarvam.SarvamLLMTranslator().translate(["a", "b"], "mr", "en") == ["x", "x"]
    assert sizes[:2] == [2, 2] and sizes[2:] == [1, 1]


def test_gives_a_clear_error_when_a_single_line_never_fits(monkeypatch):
    monkeypatch.setattr(sarvam, "request_with_retry", lambda *a, **k: reply("length"))
    with pytest.raises(ProviderError, match="running out of output space"):
        sarvam.SarvamLLMTranslator().translate(["a"], "mr", "en")
