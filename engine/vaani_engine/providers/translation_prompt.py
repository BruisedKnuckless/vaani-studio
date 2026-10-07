"""The translation instructions shared by every LLM translator (Claude, OpenAI, on-device)."""

from __future__ import annotations

import json
import re

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi, in Devanagari script",
    "mr": "Marathi, in Devanagari script",
    "hinglish": ("Hinglish: everyday Hindi mixed with English, written in Roman (Latin) script the way "
                 "people text in India, e.g. \"aaj meeting thodi late start hogi\""),
    "auto": "the language it is written in (detect it)",
}

# Comfortable speaking rates used to turn a time slot into a syllable budget.
SYLLABLES_PER_SECOND = {"en": 4.2, "hi": 5.5, "mr": 5.5, "hinglish": 5.0}

SCHEMA = {
    "type": "object",
    "properties": {"translations": {"type": "array", "items": {"type": "string"}}},
    "required": ["translations"],
    "additionalProperties": False,
}


STYLE = {
    "en": (
        "Write the English a skilled professional translator would deliver for a documentary voice-over or "
        "a polished talk: fluent, idiomatic, grammatically flawless, and articulate, in an adult, confident "
        "register. Natural spoken English with contractions is fine; stiff, bookish or childish phrasing is not."
    ),
    "hi": "Write natural, fluent spoken Hindi as an educated speaker would say it, not bookish शुद्ध Hindi; "
          "English words that Hindi speakers normally use can stay.",
    "mr": "Write natural, fluent spoken Marathi as an educated speaker would say it, not bookish Marathi; "
          "English words that Marathi speakers normally use can stay.",
    "hinglish": "Write natural Hinglish as people actually say it.",
}


def system_prompt(src: str, tgt: str, dubbing: bool) -> str:
    target = LANGUAGE_NAMES.get(tgt, tgt)
    source = LANGUAGE_NAMES.get(src, src)
    lines = [
        "You are an expert translator of spoken Indian-language content.",
        f"Source: {source}. Target: {target}.",
        STYLE.get(tgt, ""),
        "Translate the meaning, not the words. Restructure sentences freely so they read as if originally "
        "written in the target language. Never produce word-for-word calques such as \"what happens is that\", "
        "\"that is\", \"so actually what is it\" or \"in short what\".",
        "Spoken source text is full of discourse fillers and false starts (e.g. तर, म्हणजे, काय होतं, "
        "actually, म्हणजे थोडक्यात). Drop them or fold them into fluent phrasing unless they carry meaning.",
        "Render repetition and redundancy the way a good writer would: keep deliberate emphasis once, in a "
        "natural form (\"slow breathing, slow breathing, slow breathing\" → \"slow, steady breathing\"), and merge "
        "doubled ideas (\"no tension, no stress\" → \"without any stress\").",
        "Plain text only: no markdown, asterisks or quotation marks around titles.",
        "The input comes from speech recognition and may contain misheard words. Infer what the speaker "
        "meant from context. Never transliterate the source into Latin script instead of translating it.",
        "The lines are consecutive parts of one recording and a sentence may run across lines. Use the whole "
        "passage for context, but keep each line's content in that line (each is voiced in its own time "
        "slot), so that every line reads well on its own and the lines flow naturally in sequence. Return "
        "exactly one translation per input line, in order.",
        "Keep names, titles (e.g. book titles), and established terms such as pranayama or Bhastrika as they "
        "are. Keep numbers as digits.",
        "If a line is already in the target language, return it polished but unchanged in meaning. If a line "
        "is empty or only a filler sound, return an empty string.",
    ]
    if tgt == "hinglish":
        lines.append("Keep English words that Indians normally say in English (phone, meeting, office) in English.")
    if dubbing:
        lines.append(
            "These lines will be spoken by a dubbing voice in the same time slot as the original. Each line "
            "has a 'max_syllables' budget. It is an upper limit, not a target: shorter is fine, so never pad a "
            "line or repeat words or phrases to fill it.")
    lines.append('Reply with JSON only: {"translations": ["...", ...]}.')
    return "\n".join(line for line in lines if line)


def user_message(texts: list[str], tgt: str, durations: list[float] | None) -> str:
    items = []
    rate = SYLLABLES_PER_SECOND.get(tgt, 4.5)
    for i, text in enumerate(texts):
        item = {"line": i + 1, "text": text}
        if durations:
            seconds = max(0.5, durations[i])
            item["seconds"] = round(seconds, 1)
            item["max_syllables"] = max(2, int(seconds * rate))
        items.append(item)
    return json.dumps({"lines": items}, ensure_ascii=False)


def parse(output: str, expected: int) -> list[str]:
    """Pull the translations list out of a model reply, tolerating code fences or extra text."""
    text = output.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    data = None
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            data = None
    if isinstance(data, dict) and isinstance(data.get("translations"), list):
        out = [str(t) for t in data["translations"]]
    else:
        start, end = text.find("["), text.rfind("]")
        out = json.loads(text[start:end + 1]) if start != -1 and end > start else []
        out = [str(t) for t in out]
    if len(out) != expected:
        raise ValueError(f"Expected {expected} translations, got {len(out)}")
    return out


def batches(texts: list[str], durations: list[float] | None, size: int = 40):
    """Split long transcripts so each request stays small; yields (offset, texts, durations)."""
    for i in range(0, len(texts), size):
        yield i, texts[i:i + size], (durations[i:i + size] if durations else None)


def strip_loops(text: str, max_repeats: int = 3) -> str:
    """Cut a degenerate repetition loop ("रोज रोज रोज …") that small local models sometimes fall into."""
    words = text.split()
    for size in range(1, 6):
        run = 1
        for i in range(size, len(words) - size + 1, size):
            if words[i:i + size] == words[i - size:i]:
                run += 1
                if run > max_repeats:
                    start = i - (run - 1) * size
                    return " ".join(words[:start + size])
            else:
                run = 1
    return text


def salvage_single(reply: str) -> str:
    """Best effort for a one-line reply that isn't valid JSON (e.g. cut off mid-array)."""
    text = reply.strip()
    match = re.search(r'"translations"\s*:\s*\[\s*"((?:[^"\\]|\\.)*)', text)
    if match:
        try:
            return json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            return match.group(1)
    return text.strip('"[]{} \n')


ENGLISH_COMMON = set("""a an the is are was were be been am i you he she it we they me my your our their his her its
this that these those to of in on at for with from by as and or but if so not no yes do does did can could will would
should must have has had there here what when where why how who which all just very more most some any one two get
go going make take come see know think say said want need like good well also only then than now out up down about
into over after before because really breathe breathing life day time people stress""".split())


def script_ratio(text: str, lo: int, hi: int) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters) if letters else 0.0


def is_in_language(text: str, lang: str) -> bool:
    """Cheap check that a translation is really in the target language, catching the
    failure where a model just spells the source out in another script."""
    text = text.strip()
    if not text:
        return True
    devanagari = script_ratio(text, 0x0900, 0x097F)
    if lang in ("hi", "mr"):
        return devanagari >= 0.5
    if lang in ("en", "hinglish"):
        if devanagari > 0.2:
            return False
        if lang == "hinglish":
            return True
        words = re.findall(r"[a-zA-Z']+", text.lower())
        if len(words) < 4:
            return True
        return sum(w in ENGLISH_COMMON for w in words) / len(words) >= 0.15
    return True


def clean_output(text: str) -> str:
    """Remove markdown emphasis an LLM may add (e.g. *Book Title*); subtitles and voices want plain text."""
    text = re.sub(r"(\*{1,2}|_{1,2})(\S(?:.*?\S)?)\1", r"\2", text)
    return " ".join(text.replace("*", "").split())
