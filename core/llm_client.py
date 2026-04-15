"""
Unified LLM client for OpenRouter (sandbox) or Anthropic (production).
Integrates with Langfuse via @observe decorator.
"""
from __future__ import annotations
import os
import json
import time
from typing import Any
from langfuse import observe, get_client

# Ordered fallback list — tried in sequence on rate limit / 404
FALLBACK_MODELS = [
    "openrouter/elephant-alpha",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "minimax/minimax-m2.5:free",
    "arcee-ai/trinity-large-preview:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "z-ai/glm-4.5-air:free",
    "qwen/qwen3-coder:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "google/gemma-4-26b-a4b-it",
]


class LLMClient:
    """
    Wraps OpenRouter (OpenAI-compatible) API with Langfuse observability.
    Model: any OpenRouter model, defaults to a free one for sandbox.
    Falls back through FALLBACK_MODELS on rate-limit errors.
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
        from openai import RateLimitError, NotFoundError

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

        models_to_try = [self.model] + [m for m in FALLBACK_MODELS if m != self.model]
        response = None

        for model_id in models_to_try:
            kwargs["model"] = model_id
            for attempt in range(3):
                try:
                    response = self._client.chat.completions.create(**kwargs)
                    if model_id != self.model:
                        print(f"Using fallback model: {model_id}")
                    break
                except NotFoundError:
                    print(f"Model {model_id} not found, trying next fallback...")
                    break  # Don't retry 404s — move to next model
                except RateLimitError:
                    if attempt == 2:
                        print(f"Model {model_id} rate limited, trying next fallback...")
                        break
                    wait = 2 ** attempt * 5  # 5s, 10s
                    print(f"Rate limited on {model_id}, retrying in {wait}s... ({attempt + 1}/3)")
                    time.sleep(wait)
            if response is not None:
                break

        if response is None:
            raise RuntimeError("All models exhausted (rate limits / not found). Try again later.")

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
