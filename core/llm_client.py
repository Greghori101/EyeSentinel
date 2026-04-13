"""
Unified LLM client for OpenRouter (sandbox) or Anthropic (production).
Integrates with Langfuse via @observe decorator.
"""
from __future__ import annotations
import os
import json
from typing import Any
from langfuse import observe, get_client


class LLMClient:
    """
    Wraps OpenRouter (OpenAI-compatible) API with Langfuse observability.
    Model: any OpenRouter model, defaults to a free one for sandbox.
    """

    def __init__(self, model: str, session_id: str):
        self.model = model
        self.session_id = session_id
        self._client = self._build_client()

    def _build_client(self):
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Add it to .env\n"
                "Get a free key at https://openrouter.ai"
            )
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    @observe(name="llm_chat", as_type="generation")
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Send messages to the LLM and return a normalized response dict.
        Logged as a GENERATION span in Langfuse via @observe.
        """
        lf = get_client()
        lf.update_current_generation(
            model=self.model,
            input=messages,
        )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                }
                for tc in msg.tool_calls
            ]

        content = msg.content or ""

        lf.update_current_generation(
            output=content,
            usage_details={
                "input": response.usage.prompt_tokens if response.usage else 0,
                "output": response.usage.completion_tokens if response.usage else 0,
            },
        )

        return {
            "content": content,
            "tool_calls": tool_calls,
            "raw_message": msg,
        }
