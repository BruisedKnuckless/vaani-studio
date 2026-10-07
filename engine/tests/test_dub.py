import numpy as np
import pytest

from vaani_engine.pipelines import dub
from vaani_engine.providers.base import Segment

SR = 48000


def segs(*spans):
    return [Segment(a, b, f"line {i}") for i, (a, b) in enumerate(spans)]


def test_time_slots_spill_into_silence_but_not_into_next_line():
    slots = dub.time_slots(segs((0, 2), (2.2, 4), (8, 9)), total=9.5)
    assert slots[0] == pytest.approx(2.15)          # stops 50 ms before the next line
    assert slots[1] == pytest.approx(1.8 + dub.SPILL_S)
    assert slots[2] == pytest.approx(1.5)           # end of file


def test_pick_reference_prefers_6_to_14_seconds_of_continuous_speech():
    s = segs((0, 2), (2.3, 6), (6.2, 9.5), (20, 21))
    start, end, text = dub.pick_reference(s)
    assert (start, end) == (0, 9.5)
    assert text == "line 0 line 1 line 2"


def test_pick_reference_falls_back_to_short_speech():
    start, end, _ = dub.pick_reference(segs((1, 3)))
    assert (start, end) == (1, 3)
    with pytest.raises(ValueError):
        dub.pick_reference([])


def test_fit_to_slot_speeds_up_within_limit():
    clip = np.random.default_rng(0).normal(0, 0.1, (1, int(2.4 * SR))).astype(np.float32)
    out, ratio = dub.fit_to_slot(clip, SR, slot=2.0)
    assert ratio == pytest.approx(1.2, abs=0.01)
    assert out.shape[-1] / SR == pytest.approx(2.0, abs=0.05)
    out, ratio = dub.fit_to_slot(clip, SR, slot=1.0)
    assert ratio == dub.MAX_STRETCH
    same, ratio = dub.fit_to_slot(clip, SR, slot=3.0)
    assert ratio == 1.0 and same is clip


def test_place_respects_slot_and_fades():
    track = np.zeros((1, SR * 3), dtype=np.float32)
    dub.place(track, np.ones((1, SR * 2), dtype=np.float32), SR, start=0.5, max_len=1.0)
    assert track[0, int(0.4 * SR)] == 0
    assert track[0, int(1.0 * SR)] == pytest.approx(1.0)
    assert track[0, int(1.6 * SR)] == 0                   # cut at the slot end
    assert track[0, int(0.5 * SR)] < 0.1                  # faded in


def test_trim_silence_keeps_speech():
    clip = np.zeros((1, SR), dtype=np.float32)
    clip[0, 20000:30000] = 0.5
    out = dub.trim_silence(clip, SR)
    assert 10000 <= out.shape[-1] <= 10000 + 2 * int(0.03 * SR) + 1


def test_clean_for_speech_drops_stage_directions():
    assert dub.clean_for_speech("Hello (laughs) *there*  [music]") == "Hello there"
