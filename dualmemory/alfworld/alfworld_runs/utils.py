import os
import sys
import openai
from openai import OpenAI
import time
import json
from api_config import get_api_model, resolve_api_for_model
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
    retry_if_exception,
)
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


def is_retryable_error(exception):
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

from typing import Dict, List, Optional, Tuple
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


Model = Literal["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", "text-davinci-003", "gpt-3.5-turbo-instruct"]

_OPENAI_CLIENTS: Dict[Tuple[str, str], OpenAI] = {}


def _get_openai_client(model_name: Optional[str] = None) -> OpenAI:
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
def get_completion(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    stop_strs: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    model_name = model or get_api_model("completion", "deepseek-ai/DeepSeek-V3")
    response = _get_openai_client(model_name).chat.completions.create(
        model=model_name, # gpt-4o
        messages=messages,
        max_tokens=max_tokens,
        stop=stop_strs,
        temperature=temperature,
    )

    return response.choices[0].message.content

@retry(
    wait=wait_random_exponential(min=1, max=60), 
    stop=stop_after_attempt(6),
    retry=retry_if_exception(is_retryable_error),
    reraise=True,
)
def get_chat(prompt: str, model: Model, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None, is_batched: bool = False) -> str:
    assert model != "text-davinci-003"
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = _get_openai_client(model).chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stop=stop_strs,
        temperature=temperature,
    )

    return response.choices[0].message.content



@retry(
    wait=wait_random_exponential(min=1, max=60), 
    stop=stop_after_attempt(6),
    retry=retry_if_exception(is_retryable_error),
    reraise=True,
)
def get_chat_jsonoutput(
    prompt: str,
    model: Optional[Model] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    stop_strs: Optional[List[str]] = None,
    is_batched: bool = False,
) -> str:
    model_name = model or get_api_model("json_chat", "deepseek-ai/DeepSeek-V3")
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = _get_openai_client(model_name).chat.completions.create(
        model=model_name,
        messages=messages,
        # max_tokens=max_tokens,
        # stop=stop_strs,
        temperature=temperature,
        response_format = { "type": "json_object" }
    )

    return response.choices[0].message.content
