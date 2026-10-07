"""Claude (Anthropic) translation: context-aware, handles Hinglish and dubbing time budgets."""

from __future__ import annotations

import json
import os

from ...config import settings
from .. import translation_prompt as tp
from ..base import ProgressFn, ProviderError, Translator, noop_progress

MODEL = os.environ.get("VAANI_CLAUDE_MODEL", "claude-opus-5-5")


class ClaudeTranslator(Translator):
    id = "claude_mt"
    name = "Claude"
    kind = "cloud"
    key_name = "anthropic"
    languages = frozenset({"auto", "en", "hi", "mr", "hinglish"})
    rank = 10
    follows_length = True

    def translate(self, texts, src, tgt, durations=None, progress: ProgressFn = noop_progress):
        import anthropic

        client = anthropic.Anthropic(api_key=settings.key("anthropic"), timeout=300.0, max_retries=3)
        system = tp.system_prompt(src, tgt, dubbing=durations is not None)
        out: list[str] = []
        for offset, batch, batch_durations in tp.batches(texts, durations, size=60):
            progress(offset / max(len(texts), 1), f"Translating {offset + 1}–{offset + len(batch)}")
            out.extend(self._request(client, system, tp.user_message(batch, tgt, batch_durations), len(batch)))
        progress(1.0, "Translated")
        return out

    def _request(self, client, system: str, user: str, expected: int) -> list[str]:
        import anthropic

        try:
            response = client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "medium",
                               "format": {"type": "json_schema", "schema": tp.SCHEMA}},
                # On a safety decline, the API re-runs the request on Anthropic's recommended fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError:
            raise ProviderError("Claude: the Anthropic API key was rejected (check it in Settings)")
        except anthropic.PermissionDeniedError as exc:
            raise ProviderError(f"Claude: permission denied ({exc.message})")
        except anthropic.RateLimitError:
            raise ProviderError("Claude: rate limited, try again shortly")
        except anthropic.BadRequestError as exc:
            raise ProviderError(f"Claude: request rejected ({exc.message})")
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Claude: API error {exc.status_code}")
        except anthropic.APIConnectionError:
            raise ProviderError("Claude: can't reach the Anthropic API (check your internet connection)")

        if response.stop_reason == "refusal":
            raise ProviderError("Claude declined to translate this text")
        if response.stop_reason == "max_tokens":
            raise ProviderError("Claude: the translation was cut off (transcript too long for one request)")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            translations = json.loads(text)["translations"]
        except (json.JSONDecodeError, KeyError, TypeError):
            translations = tp.parse(text, expected)
        if len(translations) != expected:
            raise ProviderError(f"Claude returned {len(translations)} lines for {expected}")
        return [str(t) for t in translations]
