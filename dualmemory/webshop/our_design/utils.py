import os
import time
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from api_config import get_api_model, resolve_api_for_model
from llm_metrics import record_response

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)


NON_RETRY_STATUS_CODES = {400, 401, 403, 404, 409, 422}
RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def is_retryable_error(exception: Exception) -> bool:
    if isinstance(
        exception,
        (
            PermissionDeniedError,
            AuthenticationError,
            BadRequestError,
            NotFoundError,
            ConflictError,
            UnprocessableEntityError,
        ),
    ):
        return False

    if isinstance(exception, APIStatusError):
        if exception.status_code in NON_RETRY_STATUS_CODES:
            return False
        if exception.status_code in RETRY_STATUS_CODES:
            return True

    return isinstance(
        exception,
        (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            InternalServerError,
            APIError,
        ),
    )


_OPENAI_CLIENTS: Dict[Tuple[str, str], OpenAI] = {}


def get_openai_client(model_name: Optional[str] = None) -> OpenAI:
    settings = resolve_api_for_model(model_name=model_name)
    api_key = settings.get("api_key") or ""
    api_base = settings.get("api_base") or ""

    cache_key = (api_key, api_base)
    cached = _OPENAI_CLIENTS.get(cache_key)
    if cached is not None:
        return cached

    client = OpenAI(api_key=api_key or None, base_url=api_base or None)
    _OPENAI_CLIENTS[cache_key] = client
    return client


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception(is_retryable_error),
    reraise=True,
)
def chat_one_line(
    *,
    prompt: str,
    system: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 128,
    stop: Optional[List[str]] = None,
) -> str:
    model_name = model or get_api_model("chat", "gpt-4o-mini")
    started_at = time.perf_counter()
    response = get_openai_client(model_name).chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )
    record_response(
        model=model_name,
        component="webshop_action",
        started_at=started_at,
        response=response,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("empty LLM response")
    return text.splitlines()[0].strip()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
