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


def test_pick_reference_prefers_10_to_15_seconds_of_continuous_speech():
    s = segs((0, 2), (2.3, 6), (6.2, 11.5), (11.8, 13), (30, 31))
    start, end, text = dub.pick_reference(s)
    assert (start, end) == (0, 13)
    assert text == "line 0 line 1 line 2 line 3"


def test_pick_reference_avoids_the_closing_line():
    # Two equally good stretches; the second ends the recording (a sign-off), so the first wins.
    s = segs((0, 12), (40, 41), (60, 72))
    assert dub.pick_reference(s)[:2] == (0, 12)


def test_pick_reference_falls_back_to_short_speech_and_caps_long_speech():
    assert dub.pick_reference(segs((1, 3)))[:2] == (1, 3)
    assert dub.pick_reference(segs((0, 40)))[:2] == (0, 14.5)
    with pytest.raises(ValueError):
        dub.pick_reference([])


def tsegs(*items):
    return [Segment(a, b, f"src {i}", t) for i, (a, b, t) in enumerate(items)]


def test_group_sentences_joins_a_sentence_split_across_lines():
    s = tsegs((0, 5, "Without any stress, you can't grow. I mean, if kids have no exam pressure"),
              (5.2, 9, "they won't score well."), (9.3, 12, "So we do need stress."))
    passages = dub.group_sentences(s)
    assert [p.indices for p in passages] == [[0, 1], [2]]
    assert passages[0].text.endswith("they won't score well.")
    assert passages[0].gap_after == pytest.approx(0.3)


def test_group_sentences_breaks_on_long_pauses_length_and_untranslatable_lines():
    s = tsegs((0, 2, "First part"), (6, 8, "after a long pause."), (8.2, 9, "x" * 50), (9.1, 10, "y" * 50),
              (10.2, 11, "kept."))
    passages = dub.group_sentences(s, max_chars=80, keep_original={4})
    assert [p.indices for p in passages] == [[0], [1], [2], [3], [4]]
    assert passages[-1].original_voice and passages[-1].text == ""


def test_layout_natural_never_overlaps_and_clamps_pauses():
    passages = [dub.Passage([0], "a", 1.0, 5.0, gap_after=3.0), dub.Passage([1], "b", 8.0, 9.0, gap_after=0.05),
                dub.Passage([2], "c", 9.05, 12.0)]
    end = dub.layout_natural(passages, [6.0, 2.0, 1.0])
    assert [p.placed_at for p in passages] == [1.0, 1.0 + 6.0 + dub.MAX_PAUSE_S, 8.0 + 2.0 + dub.MIN_PAUSE_S]
    for a, b in zip(passages, passages[1:]):
        assert a.placed_at + a.spoken < b.placed_at
    assert end == pytest.approx(passages[-1].placed_at + 1.0)  # no pause padded after the last one


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
    dub.place(track, np.ones((1, SR * 2), dtype=np.float32), SR, 0.5, max_len=1.0)
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
