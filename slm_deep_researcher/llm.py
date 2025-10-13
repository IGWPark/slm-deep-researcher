"""LLM client setup for OpenAI-compatible servers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

import requests
from langchain_openai import ChatOpenAI

from .config import Configuration


@lru_cache(maxsize=16)
def _list_models(base_url: str, api_key: Optional[str]) -> list[str]:
    base = (base_url or "").rstrip("/")
    if not base:
        return []
    url = f"{base}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network variability
        raise RuntimeError(f"Failed to list models from {url}: {exc}") from exc

    payload = response.json()
    models: list[str] = []
    for item in payload.get("data", []):
        model_id = item.get("id")
        if model_id:
            models.append(model_id)
    return models


def create_chat_model(
    config: Configuration,
    *,
    json_mode: bool = False,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    model_name = config.llm_model
    available: Optional[list[str]] = None
    try:
        available = _list_models(config.llm_base_url, config.llm_api_key)
    except RuntimeError as exc:
        if not model_name:
            raise
        print(f"Warning: {exc}")

    if available:
        if model_name:
            if model_name not in available:
                raise ValueError(
                    f"Configured llm_model '{model_name}' not served by provider; available={available}"
                )
        else:
            model_name = available[0]

    if not model_name:
        raise ValueError("llm_model must be set or discoverable from the provider")

    model_kwargs: Dict[str, Any] = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    extra_body: Dict[str, Any] = {}
    if config.llm_top_k is not None:
        extra_body["top_k"] = config.llm_top_k
    if config.llm_min_p is not None:
        extra_body["min_p"] = config.llm_min_p
    if config.llm_strip_thinking:
        extra_body["skip_special_tokens"] = False

    kwargs: Dict[str, Any] = {
        "model": model_name,
        "temperature": config.llm_temperature,
        "base_url": config.llm_base_url,
        "model_kwargs": model_kwargs,
    }
    if config.llm_top_p is not None:
        kwargs["top_p"] = config.llm_top_p
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if extra_body:
        kwargs["extra_body"] = extra_body
    if config.llm_api_key:
        kwargs["api_key"] = config.llm_api_key
    else:
        kwargs["api_key"] = "placeholder"
    return ChatOpenAI(**kwargs)
