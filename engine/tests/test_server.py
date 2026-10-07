import numpy as np
from fastapi.testclient import TestClient

from vaani_engine import server
from vaani_engine.config import settings


def test_studio_voices_list_recommended_first():
    voices = TestClient(server.app).get("/voices/studio").json()
    assert [v["id"] for v in voices[:2]] == ["shubh", "ishita"]
    assert all(v["recommended"] for v in voices[:2]) and not voices[2]["recommended"]


def test_voice_preview_needs_a_key_then_is_generated_once_and_cached(monkeypatch):
    client = TestClient(server.app)
    settings.update(keys={"sarvam": None})
    assert client.post("/voices/preview", json={"voice": "ishita", "language": "en"}).status_code == 409
    assert client.post("/voices/preview", json={"voice": "nobody"}).status_code == 404

    calls = []

    def fake(text, language, ref, ref_text, max_duration=None, voice=None):
        calls.append(voice)
        return np.zeros((1, 2400), dtype=np.float32), 24000

    monkeypatch.setattr(server.registry.get("sarvam_bulbul"), "synthesize", fake)
    settings.update(keys={"sarvam": "test"})
    try:
        first = client.post("/voices/preview", json={"voice": "ishita", "language": "hi"}).json()
        second = client.post("/voices/preview", json={"voice": "ishita", "language": "hi"}).json()
    finally:
        settings.update(keys={"sarvam": None})
    assert first["cached"] is False and second["cached"] is True
    assert first["path"].endswith("previews/ishita_hi.wav") and calls == ["ishita"]
