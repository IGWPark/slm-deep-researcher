"""Configuration loaded from environment variables and LangGraph."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from typing import Any, Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig


load_dotenv()


class Configuration(BaseModel):
    """All settings for the research agent, loaded from .env or LangGraph config."""

    max_web_research_loops: int = Field(default=3)
    llm_model: Optional[str] = Field(default=None)
    llm_base_url: str = Field(default="http://localhost:8000/v1")
    llm_api_key: Optional[str] = Field(default=None)
    llm_temperature: float = Field(default=0.0)
    llm_top_p: Optional[float] = Field(default=None)
    llm_top_k: Optional[int] = Field(default=None)
    llm_min_p: Optional[float] = Field(default=None)
    llm_max_input_tokens: int = Field(default=16384)
    llm_max_output_tokens_query: int = Field(default=128)
    llm_max_output_tokens_summary: int = Field(default=2048)
    llm_max_output_tokens_reflection: int = Field(default=256)
    llm_strip_thinking: bool = Field(default=True)
    fetch_full_page: bool = Field(default=True)
    use_tool_calling: Literal["json", "tools"] = Field(default="json")
    ddgs_region: str = Field(default="us-en")
    search_max_results: int = Field(default=3)
    max_tokens_per_source: int = Field(default=1000)

    @classmethod
    def _collect_env_overrides(cls) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        for field_name in cls.model_fields:
            env_key = field_name.upper()
            if env_key in os.environ and os.environ[env_key] != "":
                overrides[field_name] = os.environ[env_key]
        return overrides

    @classmethod
    def from_env(cls) -> "Configuration":
        """Loads settings from environment variables only."""
        return cls(**cls._collect_env_overrides())

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Loads settings from both environment and LangGraph runtime config."""
        configurable: dict[str, Any] = (
            config.get("configurable", {}) if config else {}
        )
        overrides = cls._collect_env_overrides()
        for key, value in configurable.items():
            if value is not None and key in cls.model_fields:
                overrides[key] = value
        return cls(**overrides)
