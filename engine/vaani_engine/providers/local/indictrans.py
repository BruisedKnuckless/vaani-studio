"""AI4Bharat IndicTrans2 (distilled 200M): English ↔ Hindi/Marathi on this Mac."""

from __future__ import annotations

import gc

from ...models_manager import models
from ..base import ProgressFn, Translator, noop_progress
from .torch_utils import torch_device

CODES = {"en": "eng_Latn", "hi": "hin_Deva", "mr": "mar_Deva"}


class IndicTrans2(Translator):
    id = "indictrans2"
    name = "IndicTrans2"
    kind = "local"
    license = "MIT"
    languages = frozenset({"en", "hi", "mr"})
    model_ids = ("indictrans2-en-indic", "indictrans2-indic-en")
    python_modules = ("IndicTransToolkit", "transformers")
    rank = 10
    heavy = True

    def __init__(self):
        self._loaded: dict[str, tuple] = {}
        self._processor = None

    def models_for(self, src=None, tgt=None):
        if src == "en":
            return ("indictrans2-en-indic",)
        if tgt == "en" and src in ("hi", "mr"):
            return ("indictrans2-indic-en",)
        return self.model_ids  # unknown source, or Hindi ↔ Marathi (pivots through English)

    def unload(self):
        self._loaded.clear()
        gc.collect()

    def _model(self, model_id: str):
        if model_id not in self._loaded:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            path = str(models.download(model_id))
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(path, trust_remote_code=True)
            device = torch_device()
            try:
                model = model.to(device)
            except Exception:
                device = "cpu"
            self._loaded[model_id] = (tokenizer, model.eval(), device)
        return self._loaded[model_id]

    def _processor_(self):
        if self._processor is None:
            from IndicTransToolkit.processor import IndicProcessor

            self._processor = IndicProcessor(inference=True)
        return self._processor

    def _run(self, texts: list[str], src: str, tgt: str, progress: ProgressFn) -> list[str]:
        import torch

        model_id = "indictrans2-en-indic" if src == "en" else "indictrans2-indic-en"
        tokenizer, model, device = self._model(model_id)
        ip = self._processor_()
        out: list[str] = []
        for i in range(0, len(texts), 16):
            progress(i / max(len(texts), 1), f"Translating {i + 1}–{min(i + 16, len(texts))}")
            batch = [t if t.strip() else "." for t in texts[i:i + 16]]
            prepared = ip.preprocess_batch(batch, src_lang=CODES[src], tgt_lang=CODES[tgt])
            inputs = tokenizer(prepared, truncation=True, padding="longest", return_tensors="pt",
                               return_attention_mask=True).to(device)
            with torch.no_grad():
                generated = model.generate(**inputs, use_cache=False, min_length=0, max_length=256,
                                           num_beams=4, num_return_sequences=1)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)
            out.extend(ip.postprocess_batch(decoded, lang=CODES[tgt]))
        return [o if t.strip() else "" for o, t in zip(out, texts)]

    def translate(self, texts, src, tgt, durations=None, progress: ProgressFn = noop_progress):
        if src == tgt:
            return list(texts)
        if src != "en" and tgt != "en":
            english = self._run(texts, src, "en", lambda f, m="": progress(f / 2, m))
            return self._run(english, "en", tgt, lambda f, m="": progress(0.5 + f / 2, m))
        return self._run(texts, src, tgt, progress)
