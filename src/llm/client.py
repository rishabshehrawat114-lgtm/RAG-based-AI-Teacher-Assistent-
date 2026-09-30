"""
src/llm/client.py
=================
LLM Client integration module for RAG AI Teacher Assistant.

Handles secure communication with Large Language Models (e.g., OpenAI gpt-4o-mini, gpt-4o)
for grounded educational answer synthesis. Implements robust error handling, API key
validation, and fallback behavior.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Custom exception raised when LLM generation, connection, or authentication fails."""
    pass


class LLMClient:
    """
    Client for interacting with LLM APIs for educational question answering.

    Supports OpenAI API models (e.g., gpt-4o-mini, gpt-4o) and OpenAI-compatible endpoints.
    Provides clear error handling for missing/invalid keys, rate limits, and network issues.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
    ) -> None:
        """
        Initializes the LLM client.

        Args:
            api_key: OpenAI API key. If None, checks OPENAI_API_KEY from config/environment.
            model_name: Model identifier (e.g., 'gpt-4o-mini').
            base_url: Optional base URL for OpenAI-compatible endpoints.
            temperature: Sampling temperature for generation (default: 0.2).
        """
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = (OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")).strip()
        self.model_name = (model_name or DEFAULT_LLM_MODEL).strip()
        self.base_url = (base_url or OPENAI_BASE_URL or os.getenv("OPENAI_BASE_URL", "")).strip() or None
        self.temperature = float(temperature if temperature is not None else DEFAULT_LLM_TEMPERATURE)

        self._client: Any = None
        self._quota_exhausted: bool = False
        self._quota_error_msg: str = ""
        if self.is_configured():
            self._init_client()

    def is_configured(self) -> bool:
        """Checks whether a non-empty API key is present."""
        return bool(self.api_key and not self.api_key.startswith("your_"))

    def set_api_key(self, api_key: str) -> None:
        """Updates the API key and resets any cached quota exhaustion status."""
        self.api_key = (api_key or "").strip()
        self._quota_exhausted = False
        self._quota_error_msg = ""
        self._client = None
        if self.is_configured():
            self._init_client()

    def _init_client(self) -> None:
        """Initializes the underlying OpenAI SDK client."""
        try:
            import openai
            kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "max_retries": 1,
                "timeout": 12.0,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = openai.OpenAI(**kwargs)
        except Exception as exc:
            logger.warning("Failed to initialize OpenAI client: %s", exc)
            self._client = None

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generates a completion from a list of chat messages.

        Args:
            messages: List of message dicts: [{'role': 'system'|'user'|'assistant', 'content': str}]
            temperature: Optional override for sampling temperature.
            max_tokens: Optional token limit for generation.

        Returns:
            The assistant's generated response string.

        Raises:
            LLMError: If authentication fails, rate limit is exceeded, or API is unreachable.
        """
        if not messages:
            raise LLMError("Cannot generate response: messages list is empty.")

        # Re-check client initialization if key was supplied later
        if self._client is None and self.is_configured():
            self._init_client()

        if not self.is_configured() or self._client is None:
            raise LLMError(
                "OpenAI API key is missing or not configured. "
                "Please set OPENAI_API_KEY in your .env file or enter your key in the web interface."
            )

        # Fast-fail if credentials previously returned credit exhaustion in current session
        if self._quota_exhausted:
            raise LLMError(self._quota_error_msg)

        temp = self.temperature if temperature is None else float(temperature)

        try:
            import openai

            params: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temp,
            }
            if max_tokens:
                params["max_tokens"] = max_tokens

            response = self._client.chat.completions.create(**params)
            choice = response.choices[0]
            answer = choice.message.content or ""
            return answer.strip()

        except openai.AuthenticationError as exc:
            err_msg = (
                f"OpenAI Authentication Failed: Invalid or expired API key. "
                f"Please verify your OPENAI_API_KEY. Details: {exc}"
            )
            self._quota_exhausted = True
            self._quota_error_msg = err_msg
            raise LLMError(err_msg) from exc
        except openai.RateLimitError as exc:
            err_msg = (
                f"OpenAI Rate Limit Exceeded: Your account has hit a quota or rate limit. "
                f"Please check your OpenAI billing plan. Details: {exc}"
            )
            exc_str = str(exc).lower()
            if "insufficient_quota" in exc_str or "credit_balance_exhausted" in exc_str:
                self._quota_exhausted = True
                self._quota_error_msg = err_msg
            raise LLMError(err_msg) from exc
        except openai.APIConnectionError as exc:
            raise LLMError(
                f"OpenAI Connection Error: Unable to reach OpenAI servers. "
                f"Please check your internet connection or base URL. Details: {exc}"
            ) from exc
        except openai.APITimeoutError as exc:
            raise LLMError(
                f"OpenAI Timeout: The request to the LLM timed out. Please try again. Details: {exc}"
            ) from exc
        except openai.BadRequestError as exc:
            raise LLMError(
                f"OpenAI Bad Request: Invalid model or message parameter. Details: {exc}"
            ) from exc
        except Exception as exc:
            raise LLMError(
                f"LLM Generation Failed: An unexpected error occurred: {exc}"
            ) from exc
