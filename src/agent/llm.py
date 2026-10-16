"""Chat-model factory with a provider switch from .env.

LLM_PROVIDER = anthropic (default) | openai | ollama
MODEL_NAME / BATCH_MODEL_NAME override the per-provider defaults below.
LLM_API_KEY (falls back to ANTHROPIC_API_KEY / OPENAI_API_KEY); not needed for ollama or for
an OpenAI-compatible server on localhost.
OPENAI_BASE_URL: any OpenAI-compatible API (Groq, OpenRouter, Gemini, vLLM, LM Studio, ...).
OLLAMA_BASE_URL (default http://localhost:11434), OLLAMA_NUM_CTX (default 16384).

Notes:
- Claude Sonnet 5.5 rejects non-default temperature and forced tool use, so no sampling
  parameter is sent to Anthropic and structured output is parsed from text.
- Ollama's default context window (2-4k tokens) would silently truncate the agent's
  prompts, so num_ctx is raised explicitly.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Optional
from urllib.parse import urlparse

DEFAULT_MODELS = {"anthropic": ("claude-sonnet-5-5", "claude-haiku-4-5-20251001"),
                  "openai": ("gpt-4.1-mini", "gpt-4.1-mini"),
                  "ollama": ("qwen2.5:7b", "qwen2.5:7b")}
FREE_PROVIDERS = {"ollama"}


def _env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass


def is_local_url(url: Optional[str]) -> bool:
    host = urlparse(url or "").hostname or ""
    return host in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or host.endswith(".local")


def llm_settings(purpose: str = "default") -> dict[str, Any]:
    _env()
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    main, batch = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["anthropic"])
    model = (os.getenv("BATCH_MODEL_NAME") or batch) if purpose == "batch" else (os.getenv("MODEL_NAME") or main)
    key = os.getenv("LLM_API_KEY") or {"anthropic": os.getenv("ANTHROPIC_API_KEY"),
                                       "openai": os.getenv("OPENAI_API_KEY")}.get(provider)
    base_url = {"ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "openai": os.getenv("OPENAI_BASE_URL") or None}.get(provider)
    return {"provider": provider, "model": model, "api_key": key, "base_url": base_url}


def needs_api_key(settings: dict[str, Any]) -> bool:
    if settings["provider"] == "ollama":
        return False
    if settings["provider"] == "openai" and is_local_url(settings.get("base_url")):
        return False
    return True


def check_ollama(model: str, base_url: str, timeout: float = 5.0) -> None:
    """Raise RuntimeError with the fix if the Ollama server is down or the model is not pulled."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=timeout) as r:
            names = [m.get("name", "") for m in json.load(r).get("models", [])]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Ollama is not reachable at {base_url} ({type(e).__name__}). "
                           "Start it with `ollama serve` (or set OLLAMA_BASE_URL).") from e
    wanted = model if ":" in model else model + ":latest"
    if not any(n == wanted or n == model for n in names):
        raise RuntimeError(f"Model {model!r} is not pulled on {base_url}. Run: ollama pull {model}"
                           f"  (available: {', '.join(names) or 'none'})")


def get_llm(provider: str | None = None, model: str | None = None, purpose: str = "default",
            max_tokens: int = 4096, timeout: float = 300.0, json_mode: bool = False):
    """Return a LangChain chat model, or raise RuntimeError if it cannot be configured.
    json_mode asks the backend for JSON-only output where supported (Ollama format=json)."""
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
        base_url = s["base_url"] if s["provider"] == "openai" else os.getenv("OPENAI_BASE_URL")
        kwargs = {"model": model, "api_key": s["api_key"] or "not-needed", "max_tokens": max_tokens,
                  "timeout": timeout, "max_retries": 2, "temperature": 0}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise RuntimeError("pip install langchain-ollama to use LLM_PROVIDER=ollama") from e
        base_url = s["base_url"] if s["provider"] == "ollama" else os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=0,
                          num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "16384")), num_predict=max_tokens,
                          format="json" if json_mode else None, keep_alive="30m",
                          client_kwargs={"timeout": timeout})
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
