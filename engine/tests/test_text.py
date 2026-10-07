import json

import pytest

from vaani_engine.formats import subtitles
from vaani_engine.pipelines import vad
from vaani_engine.providers import translation_prompt as tp
from vaani_engine.providers.base import Segment

SEGS = [Segment(0.0, 2.5, "नमस्ते, आप कैसे हैं?", "Hello, how are you?"),
        Segment(3661.25, 3663.0, "ठीक हूँ", "I'm fine")]


def test_srt_format():
    srt = subtitles.to_srt(SEGS)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,500\nनमस्ते")
    assert "2\n01:01:01,250 --> 01:01:03,000\nठीक हूँ" in srt


def test_vtt_uses_translation_field_and_dots():
    vtt = subtitles.to_vtt(SEGS, "translation")
    assert vtt.startswith("WEBVTT")
    assert "01:01:01.250 --> 01:01:03.000\nI'm fine" in vtt


def test_long_lines_wrap():
    seg = Segment(0, 5, "word " * 30)
    body = subtitles.to_srt([seg]).splitlines()[2:]
    assert all(len(line) <= subtitles.MAX_LINE for line in body if line)


def test_empty_segments_are_skipped_and_numbering_stays_continuous():
    segs = [Segment(0, 1, "a"), Segment(1, 2, " "), Segment(2, 3, "b")]
    assert "2\n00:00:02,000 --> 00:00:03,000\nb" in subtitles.to_srt(segs)


def test_prompt_parse_handles_fences_and_plain_arrays():
    assert tp.parse('```json\n{"translations": ["a", "b"]}\n```', 2) == ["a", "b"]
    assert tp.parse('Sure! ["x"]', 1) == ["x"]
    with pytest.raises(ValueError):
        tp.parse('{"translations": ["a"]}', 2)


def test_dubbing_message_includes_syllable_budget():
    msg = json.loads(tp.user_message(["hello"], "en", [2.0]))
    line = msg["lines"][0]
    assert line["seconds"] == 2.0 and line["max_syllables"] == int(2.0 * tp.SYLLABLES_PER_SECOND["en"])
    assert "max_syllables" in tp.system_prompt("hi", "en", dubbing=True)
    assert "max_syllables" not in tp.system_prompt("hi", "en", dubbing=False)


def test_batches_cover_everything():
    texts = [str(i) for i in range(95)]
    seen = [t for _, batch, _ in tp.batches(texts, None, size=40) for t in batch]
    assert seen == texts


def test_vad_grouping_merges_close_regions_and_splits_long_ones():
    regions = [(0.0, 1.0), (1.2, 2.0), (5.0, 6.0), (10.0, 40.0)]
    grouped = vad.group_regions(regions, max_len=12.0, max_gap=0.6)
    assert grouped[0] == (0.0, 2.0)
    assert grouped[1] == (5.0, 6.0)
    long_parts = grouped[2:]
    assert len(long_parts) == 3 and all(e - s <= 12.0 + 1e-9 for s, e in long_parts)
    assert long_parts[0][0] == 10.0 and abs(long_parts[-1][1] - 40.0) < 1e-9


def test_strip_loops_cuts_degenerate_repetition():
    assert tp.strip_loops("तुम्हाला हे दररोज रोज रोज रोज रोज रोज रोज") == "तुम्हाला हे दररोज रोज"
    assert tp.strip_loops("a b c a b c a b c a b c a b c") == "a b c"
    assert tp.strip_loops("slow breathing, slow breathing, slow breathing") == "slow breathing, slow breathing, slow breathing"


def test_marathi_hindi_heuristic():
    from vaani_engine.providers.local.asr import marathi_score

    marathi = "तर तुम्ही परीक्षेत चांगले मार्क्स मिळवू शकता, जर आयुष्यात ताण नसेल. आपण नेहमी म्हणतो"
    hindi = "अगर जीवन में तनाव नहीं है तो आप परीक्षा में अच्छे अंक ला सकते हैं"
    assert marathi_score(marathi) > 0.02
    assert marathi_score(hindi) < 0


def test_salvage_single_line_from_broken_json():
    assert tp.salvage_single('{"translations": ["In pranayama, breathe slowly') == "In pranayama, breathe slowly"
    assert tp.salvage_single('{"translations": ["He said \\"hi\\""') == 'He said "hi"'
    assert tp.salvage_single('"Just text"') == "Just text"
