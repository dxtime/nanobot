"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations

from typing import Any

import json_repair
from openai import AsyncOpenAI
from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class CustomProvider(LLMProvider):

    def __init__(self, api_key: str = "no-key", api_base: str = "http://localhost:8000/v1", default_model: str = "default"):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self._current_model = default_model
        self._available_models: list[str] = []
        self._models_loaded = False
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def _fetch_models(self) -> list[str]:
        """Fetch available models from the API endpoint."""
        try:
            response = await self._client.models.list()
            models = [m.id for m in response.data]
            logger.info("CustomProvider: fetched {} models from {}", len(models), self.api_base)
            return models
        except Exception as e:
            logger.warning("CustomProvider: failed to fetch models: {}", e)
            return []

    async def _ensure_models_loaded(self) -> None:
        """Load models on first access if not already loaded."""
        if not self._models_loaded:
            self._available_models = await self._fetch_models()
            self._models_loaded = True

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self._current_model,
            "messages": self._sanitize_empty_content(messages),
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        try:
            return self._parse(await self._client.chat.completions.create(**kwargs))
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(id=tc.id, name=tc.function.name,
                            arguments=json_repair.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        u = response.usage
        return LLMResponse(
            content=msg.content, tool_calls=tool_calls, finish_reason=choice.finish_reason or "stop",
            usage={"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens} if u else {},
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )

    def get_default_model(self) -> str:
        return self.default_model

    async def list_models(self) -> list[str] | None:
        await self._ensure_models_loaded()
        return self._available_models.copy()

    def get_current_model(self) -> str:
        return self._current_model

    def set_model(self, model_name: str) -> bool:
        if self._models_loaded:
            if model_name in self._available_models:
                self._current_model = model_name
                logger.info("CustomProvider: switched to model {}", model_name)
                return True
            return False
        if model_name:
            self._current_model = model_name
            logger.info("CustomProvider: set model to {} (not verified)", model_name)
            return True
        return False

    async def refresh_models(self) -> bool:
        try:
            self._available_models = await self._fetch_models()
            self._models_loaded = True
            return True
        except Exception:
            return False
