from fakes import make_registry


def kinds(plan):
    return {s.stage: (s.provider_id, s.status) for s in plan.steps}


def test_local_mode_uses_only_on_device_providers():
    plan = make_registry({"sarvam": "k", "anthropic": "k"}).resolve_plan("dub", "hi", "en", "local")
    assert all(s.kind == "local" for s in plan.steps)
    assert not plan.uploads and not plan.blocking


def test_cloud_mode_with_keys_uses_cloud_where_it_exists():
    plan = make_registry({"sarvam": "k", "anthropic": "k"}).resolve_plan("dub", "hi", "en", "cloud")
    steps = kinds(plan)
    assert steps["asr"] == ("cloud_asr", "ready")
    assert steps["translate"] == ("llm_mt", "ready")
    assert steps["tts"] == ("cloud_tts", "ready")
    assert "separate" not in steps  # voice-only dubs skip separation
    assert plan.uploads and not plan.blocking


def test_cloud_mode_without_key_blocks_and_offers_local_alternative():
    plan = make_registry({}).resolve_plan("transcribe", "mr", None, "cloud")
    step = plan.steps[0]
    assert step.status == "missing_key"
    assert step.local_alternative == "local_asr"
    assert plan.blocking


def test_auto_prefers_cloud_for_marathi_asr_when_key_present():
    registry = make_registry({"sarvam": "k"})
    assert kinds(registry.resolve_plan("transcribe", "mr", None, "auto"))["asr"][0] == "cloud_asr"
    assert kinds(registry.resolve_plan("transcribe", "en", None, "auto"))["asr"][0] == "local_asr"


def test_auto_without_keys_runs_everything_locally():
    plan = make_registry({}).resolve_plan("dub", "mr", "en", "auto")
    assert all(s.kind == "local" for s in plan.steps)
    assert not plan.blocking


def test_auto_records_fallback_on_the_other_side():
    plan = make_registry({"sarvam": "k"}).resolve_plan("transcribe", "mr", None, "auto")
    assert plan.steps[0].provider_id == "cloud_asr"
    assert plan.steps[0].fallback_id == "local_asr"


def test_offline_forces_local_even_in_cloud_mode_and_ignores_cloud_overrides():
    registry = make_registry({"sarvam": "k", "anthropic": "k"}, offline=True)
    plan = registry.resolve_plan("dub", "hi", "en", "cloud", overrides={"asr": "cloud_asr"})
    assert all(s.kind == "local" for s in plan.steps)
    assert not plan.uploads


def test_per_step_override_mixes_modes():
    plan = make_registry({"anthropic": "k"}).resolve_plan(
        "transcribe", "hi", "en", "local", overrides={"translate": "llm_mt"})
    steps = kinds(plan)
    assert steps["asr"][0] == "local_asr"
    assert steps["translate"][0] == "llm_mt"
    assert plan.uploads


def test_override_to_cloud_without_key_blocks():
    plan = make_registry({}).resolve_plan("transcribe", "hi", "en", "local", overrides={"translate": "llm_mt"})
    assert kinds(plan)["translate"] == ("llm_mt", "missing_key")
    assert plan.blocking


def test_enhance_has_no_cloud_option():
    registry = make_registry({"sarvam": "k", "anthropic": "k"})
    assert not registry.cloud_available_for("enhance")
    assert registry.cloud_available_for("dub")
    plan = registry.resolve_plan("enhance", mode="cloud")
    assert {s.status for s in plan.steps} <= {"local_only", "ready"}
    assert not plan.uploads


def test_translate_step_only_when_target_differs():
    registry = make_registry({})
    assert [s.stage for s in registry.resolve_plan("transcribe", "hi", None).steps] == ["asr"]
    assert [s.stage for s in registry.resolve_plan("transcribe", "hi", "hi").steps] == ["asr"]
    assert [s.stage for s in registry.resolve_plan("transcribe", "hi", "en").steps] == ["asr", "translate"]


def test_hinglish_translation_has_no_local_option_so_auto_needs_a_key():
    # The fake local MT doesn't speak Hinglish; only the LLM does.
    assert make_registry({}).resolve_plan("transcribe", "en", "hinglish", "auto").blocking
    assert not make_registry({"anthropic": "k"}).resolve_plan("transcribe", "en", "hinglish", "auto").blocking


def _voice_registry(keys):
    from fakes import FakeCloudTTS

    registry = make_registry(keys)
    clone, studio = FakeCloudTTS(), FakeCloudTTS()
    clone.id, clone.name = "sarvam_tts", "Sarvam voice clone"
    studio.id, studio.name, studio.rank = "sarvam_bulbul", "Sarvam studio voice", 50
    registry.register(clone)
    registry.register(studio)
    return registry


def test_auto_prefers_cloud_voice_when_key_present():
    plan = _voice_registry({"sarvam": "k"}).resolve_plan("dub", "mr", "en", "auto")
    assert kinds(plan)["tts"][0] in ("sarvam_tts", "cloud_tts")
    assert kinds(_voice_registry({}).resolve_plan("dub", "mr", "en", "auto"))["tts"][0] == "local_tts"


def test_voice_option_picks_engine_and_survives_mode_changes():
    registry = _voice_registry({"sarvam": "k"})
    for mode in ("local", "cloud", "auto"):
        plan = registry.resolve_plan("dub", "mr", "en", mode, options={"voice": "sarvam_studio"})
        assert kinds(plan)["tts"] == ("sarvam_bulbul", "ready")
    plan = registry.resolve_plan("dub", "mr", "en", "cloud", options={"voice": "local"})
    assert kinds(plan)["tts"][0] == "local_tts"


def test_sarvam_voice_without_key_blocks_with_local_alternative():
    plan = _voice_registry({}).resolve_plan("dub", "mr", "en", "local", options={"voice": "sarvam_clone"})
    step = next(s for s in plan.steps if s.stage == "tts")
    assert step.status == "missing_key" and step.local_alternative == "local_tts"


def test_voice_match_is_dropped_for_cloud_voices():
    registry = _voice_registry({"sarvam": "k"})
    from fakes import FakeLocalTTS
    from vaani_engine.providers.base import Stage, VoiceConverter

    class FakeVC(VoiceConverter):
        id, name, kind = "fake_vc", "Fake VC", "local"
        def convert(self, audio, sr, ref_audio_path):
            return audio, sr

    registry.register(FakeVC())
    cloud = registry.resolve_plan("dub", "mr", "en", "cloud", options={"voice_match": True})
    local = registry.resolve_plan("dub", "mr", "en", "local", options={"voice_match": True})
    assert "voice_match" not in [s.stage for s in cloud.steps]
    assert "voice_match" in [s.stage for s in local.steps]


def test_dub_separates_only_when_the_background_is_wanted():
    registry = make_registry({})
    assert "separate" not in [s.stage for s in registry.resolve_plan("dub", "mr", "en", "local").steps]
    with_bed = registry.resolve_plan("dub", "mr", "en", "cloud", options={"keep_background": True})
    assert with_bed.steps[0].stage == "separate" and with_bed.steps[0].status == "local_only"
