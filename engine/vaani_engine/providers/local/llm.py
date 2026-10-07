"""On-device LLM (MLX) for Hinglish and dubbing-aware translation."""

from __future__ import annotations

import gc
import threading

from ...models_manager import models
from .. import translation_prompt as tp
from ..base import ProgressFn, Translator, noop_progress


class _LocalLLM:
    """One on-device LLM in memory at a time."""

    _lock = threading.Lock()
    _model = None
    _tokenizer = None
    _model_id: str | None = None

    @classmethod
    def load(cls, model_id: str):
        with cls._lock:
            if cls._model is None or cls._model_id != model_id:
                from mlx_lm import load

                cls._model = cls._tokenizer = None
                gc.collect()
                cls._model, cls._tokenizer = load(str(models.download(model_id)))
                cls._model_id = model_id
        return cls._model, cls._tokenizer

    @classmethod
    def unload(cls):
        with cls._lock:
            cls._model = cls._tokenizer = cls._model_id = None
        gc.collect()

    @classmethod
    def chat(cls, model_id: str, system: str, user: str, max_tokens: int = 2048) -> str:
        from mlx_lm import generate

        model, tokenizer = cls.load(model_id)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


class LocalLLMTranslator(Translator):
    id = "local_llm"
    name = "Qwen3 4B (on-device LLM)"
    kind = "local"
    license = "Apache-2.0"
    languages = frozenset({"en", "hi", "mr", "hinglish"})
    follows_length = True
    model_ids = ("local-llm",)
    python_modules = ("mlx_lm",)
    heavy = True

    def rank_for(self, src, tgt):
        return 10 if "hinglish" in (src, tgt) else 30

    def unload(self):
        _LocalLLM.unload()

    def translate(self, texts, src, tgt, durations=None, progress: ProgressFn = noop_progress):
        system = tp.system_prompt(src, tgt, dubbing=durations is not None)
        out: list[str] = []
        for offset, batch, batch_durations in tp.batches(texts, durations, size=15):
            progress(offset / max(len(texts), 1), f"Translating {offset + 1}–{offset + len(batch)}")
            out.extend(self._translate_batch(system, batch, tgt, batch_durations))
        progress(1.0, "Translated")
        return out

    def _translate_batch(self, system, batch, tgt, durations):
        reply = _LocalLLM.chat(self.model_ids[0], system, tp.user_message(batch, tgt, durations),
                               max_tokens=200 + 120 * len(batch))
        try:
            return [tp.strip_loops(t) for t in tp.parse(reply, len(batch))]
        except (ValueError, Exception):
            if len(batch) == 1:
                return [tp.strip_loops(tp.salvage_single(reply))]
            # Small models occasionally merge lines; retry one line at a time.
            return [self._translate_batch(system, [t], tgt, [durations[i]] if durations else None)[0]
                    for i, t in enumerate(batch)]


class LocalLLMLargeTranslator(LocalLLMTranslator):
    """Gemma 3 12B: noticeably better at Marathi and Hindi than the 4B model, and better at
    making sense of imperfect transcripts. Needs ~8 GB of memory while translating."""

    id = "local_llm_large"
    name = "Gemma 3 12B (on-device LLM)"
    license = "Gemma terms of use"
    model_ids = ("local-llm-large",)

    def rank_for(self, src, tgt):
        return 10 if "hinglish" not in (src, tgt) else 20
