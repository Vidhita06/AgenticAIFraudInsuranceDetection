"""Chat-model factory with a provider switch from .env.

LLM_PROVIDER = anthropic (default) | openai | ollama
MODEL_NAME (default claude-sonnet-5-5), BATCH_MODEL_NAME (default claude-haiku-4-5-20251001)
LLM_API_KEY (falls back to ANTHROPIC_API_KEY / OPENAI_API_KEY)

No sampling parameters are set: Claude Sonnet 5.5 rejects non-default temperature.
Structured output is parsed from text (not forced tool use, which Sonnet 5.5 rejects).
"""
from __future__ import annotations

import os
from typing import Optional

DEFAULT_MODELS = {"anthropic": ("claude-sonnet-5-5", "claude-haiku-4-5-20251001"),
                  "openai": ("gpt-4.1", "gpt-4.1-mini"), "ollama": ("llama3.1", "llama3.1")}


def _env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass


def llm_settings(purpose: str = "default") -> dict[str, Optional[str]]:
    _env()
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    main, batch = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["anthropic"])
    model = (os.getenv("BATCH_MODEL_NAME") or batch) if purpose == "batch" else (os.getenv("MODEL_NAME") or main)
    key = os.getenv("LLM_API_KEY") or {"anthropic": os.getenv("ANTHROPIC_API_KEY"),
                                       "openai": os.getenv("OPENAI_API_KEY")}.get(provider)
    return {"provider": provider, "model": model, "api_key": key}


def get_llm(provider: str | None = None, model: str | None = None, purpose: str = "default",
            max_tokens: int = 4096, timeout: float = 120.0):
    """Return a LangChain chat model, or raise RuntimeError if it cannot be configured."""
    s = llm_settings(purpose)
    provider = (provider or s["provider"]).lower()
    model = model or s["model"]
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = {"model": model, "max_tokens": max_tokens, "timeout": timeout, "max_retries": 2}
        if s["api_key"]:
            kwargs["api_key"] = s["api_key"]
        return ChatAnthropic(**kwargs)
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise RuntimeError("pip install langchain-openai to use LLM_PROVIDER=openai") from e
        return ChatOpenAI(model=model, api_key=s["api_key"], max_tokens=max_tokens, timeout=timeout)
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise RuntimeError("pip install langchain-ollama to use LLM_PROVIDER=ollama") from e
        return ChatOllama(model=model, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    raise RuntimeError(f"unknown LLM_PROVIDER {provider!r}")


def message_text(message) -> str:
    """Text of a chat message whose content may be a string or a list of blocks
    (thinking / text / tool_use)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)
