"""Helpers shared by cloud providers."""

from __future__ import annotations

import io
import time

import httpx
import numpy as np
import soundfile as sf

from ..base import ProviderError

RETRY_STATUS = {429, 500, 502, 503, 504}


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Mono 16-bit WAV in memory: small and accepted by every speech API."""
    buffer = io.BytesIO()
    mono = np.asarray(audio, dtype=np.float32).reshape(-1) if np.ndim(audio) == 1 else np.asarray(audio).mean(axis=0)
    sf.write(buffer, np.clip(mono, -1, 1), sr, subtype="PCM_16", format="WAV")
    return buffer.getvalue()


def request_with_retry(method: str, url: str, *, service: str, attempts: int = 4, timeout: float = 120.0,
                       **kwargs) -> httpx.Response:
    """HTTP request with exponential backoff on rate limits and server errors."""
    delay = 1.5
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = httpx.request(method, url, timeout=timeout, **kwargs)
        except httpx.HTTPError as exc:
            last = exc
        else:
            if response.status_code < 400:
                return response
            if response.status_code not in RETRY_STATUS or attempt == attempts - 1:
                raise ProviderError(f"{service}: {_error_message(response)}")
            last = ProviderError(f"{service}: HTTP {response.status_code}")
            retry_after = response.headers.get("retry-after")
            if retry_after and retry_after.isdigit():
                delay = max(delay, float(retry_after))
        time.sleep(delay)
        delay *= 2
    raise ProviderError(f"{service} is unreachable: {last}")


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        message = error.get("message") or error.get("code")
    else:
        message = error or (body.get("detail") if isinstance(body, dict) else None)
    hint = {401: " (check the API key in Settings)", 403: " (check the API key or your plan)",
            402: " (out of credits)", 429: " (rate limited, try again shortly)"}.get(response.status_code, "")
    return f"HTTP {response.status_code}: {message}{hint}"
