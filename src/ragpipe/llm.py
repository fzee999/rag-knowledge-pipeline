"""Chat model factory: a local model via Ollama (default) or Claude via the Anthropic API.

Both are LangChain chat models, so the rest of the code (prompts, tools, structured output)
doesn't care which one is running. Pick with RAG_PROVIDER=ollama|anthropic in .env.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from ragpipe.config import get_settings

# Claude only: if a safety classifier declines a request, the API retries it server-side
# on the model Anthropic recommends for that category instead of failing.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_MODELS_WITH_DEFAULT_FALLBACKS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


def get_chat_model(effort: str | None = None) -> BaseChatModel:
    """The configured chat model.

    `effort` (low..max) sets Claude's reasoning effort; local models ignore it.
    """
    settings = get_settings()
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            # Ollama's default context is small; retrieved passages + chat history need room.
            num_ctx=settings.ollama_num_ctx,
            temperature=0,
            reasoning=_ollama_reasoning(),
        )

    from langchain_anthropic import ChatAnthropic

    kwargs: dict = {}
    if settings.llm_model in _MODELS_WITH_DEFAULT_FALLBACKS:
        kwargs = {"betas": [FALLBACK_BETA], "model_kwargs": {"fallbacks": "default"}}
    return ChatAnthropic(
        model=settings.llm_model,
        max_tokens=16000,  # covers thinking + the reply
        effort=effort or settings.llm_effort,
        **kwargs,
    )


def is_refusal(message: AIMessage) -> bool:
    """True when Claude's safety classifiers declined the request (HTTP 200, stop_reason=refusal)."""
    return message.response_metadata.get("stop_reason") == "refusal"


def _ollama_reasoning() -> bool | str | None:
    """OLLAMA_REASONING: unset = model default, true/false, or low|medium|high (gpt-oss)."""
    value = os.getenv("OLLAMA_REASONING", "").strip().lower()
    if not value:
        return None
    return {"true": True, "false": False}.get(value, value)
