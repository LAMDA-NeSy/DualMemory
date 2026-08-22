from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

from openai import OpenAI

from utils import RetryConfig, with_retries
from api_config import resolve_api_for_model
from llm_metrics import record_response


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "").strip() or "kimi-k2-0905"
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()


@dataclass(frozen=True)
class LLMConfig:
    model: str = DEFAULT_MODEL
    base_url: str | None = DEFAULT_BASE_URL
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_s: float = 120.0
    retry: RetryConfig = RetryConfig()


_CLIENT_CACHE: dict[tuple[str, str], OpenAI] = {}


def make_openai_client(
    *,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    if api_key is None and base_url is None and model_name:
        settings = resolve_api_for_model(model_name=model_name)
        api_key = settings.get("api_key") or api_key
        base_url = settings.get("api_base") or base_url

    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    url = (base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).strip()

    cache_key = (key, url)
    cached = _CLIENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    client = OpenAI(api_key=key, base_url=url or None) if url else OpenAI(api_key=key)
    _CLIENT_CACHE[cache_key] = client
    return client


def make_chat_llm(
    client: OpenAI,
    *,
    config: LLMConfig,
    system_prompt: str,
    default_stop: list[str] | None = None,
) -> Callable[[str, list[str] | None], str]:
    def _call(prompt: str, stop: list[str] | None = None) -> str:
        stop = default_stop if stop is None else stop

        def _once() -> str:
            started_at = time.perf_counter()
            resp = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                stop=stop,
                timeout=config.timeout_s,
            )
            record_response(
                model=config.model,
                component="textcraft_action",
                started_at=started_at,
                response=resp,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("empty LLM response")
            return text

        def _retryable(err: Exception) -> bool:
            # Treat almost everything as retryable: rate limits, transient network, etc.
            return True

        # Add slight jitter between retries to reduce thundering herd.
        return with_retries(
            lambda: _once(),
            retry=config.retry,
            is_retryable=_retryable,
        )

    return _call


def add_jitter_sleep(base_s: float) -> None:
    time.sleep(base_s + random.random())
