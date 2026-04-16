"""
Unified OpenRouter client using LangChain + Langfuse CallbackHandler.
This follows the challenge template pattern so token usage and costs are
tracked by Langfuse under the active session ID.
"""

from __future__ import annotations
import json
import importlib.util
import os
from typing import Any

from dotenv import load_dotenv
from langfuse import observe, get_client

load_dotenv()

# Ordered fallback list — tried in sequence on rate limit / 404
FALLBACK_MODELS = [
    # 🔥 PRIMARY (high success rate, rarely blocked)
    "anthropic/claude-sonnet-4.6",
    "deepseek/deepseek-chat-v3.2",
    "google/gemini-3.1-flash-lite",
    # ⚖️ STRONG MID (good balance, usually available)
    "z-ai/glm-5",
    "mistralai/mistral-large-3",
    "minimax/minimax-m2.5",
    # 💰 YOUR STRONG FREE MODELS (reordered by reliability)
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "qwen/qwen3-coder:free",
    # ⚡ SMALL / FAST FREE (higher uptime)
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/nemotron-nano-9b-v2:free",
    # 🧪 UNSTABLE / EXPERIMENTAL (keep last)
    "arcee-ai/trinity-large-preview:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "openrouter/elephant-alpha",
    # 🔁 NON-FREE FALLBACK (only if you allow paid implicitly)
    "google/gemma-4-26b-a4b-it",
]


class LLMClient:
    """
    Wraps OpenRouter through LangChain so Langfuse can automatically
    capture token usage, costs, latency, and session grouping.
    """

    def __init__(self, model: str, session_id: str):
        self.model = model
        self.session_id = session_id
        self._use_langchain = self._langchain_available()
        self._client = self._build_client()

    @staticmethod
    def _langchain_available() -> bool:
        return bool(importlib.util.find_spec("langchain_openai")) and bool(
            importlib.util.find_spec("langchain_core.messages")
        )

    def _build_client(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Add it to .env\n"
                "Get a free key at https://openrouter.ai"
            )
        if self._use_langchain:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                model=self.model,
                temperature=0.1,
                max_tokens=1024,
            )

        from openai import OpenAI

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
        if self._use_langchain:
            from langfuse.langchain import CallbackHandler

            langchain_messages = self._to_langchain_messages(messages)
            callback = CallbackHandler()

            bound_model = self._client
            if tools:
                bound_model = self._client.bind_tools(tools, tool_choice="auto")

            response = bound_model.invoke(
                langchain_messages,
                config={
                    "callbacks": [callback],
                    "metadata": {"langfuse_session_id": self.session_id},
                },
            )

            tool_calls = None
            if getattr(response, "tool_calls", None):
                tool_calls = [
                    {
                        "id": tc.get("id", ""),
                        "name": tc.get("name", ""),
                        "input": tc.get("args", {}),
                    }
                    for tc in response.tool_calls
                ]

            content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            usage = getattr(response, "usage_metadata", {}) or {}
            raw_message = response
        else:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            models_to_try = [self.model] + [
                m for m in FALLBACK_MODELS if m != self.model
            ]
            response = None
            for model_id in models_to_try:
                kwargs["model"] = model_id
                try:
                    response = self._client.chat.completions.create(**kwargs)
                    break
                except Exception:
                    continue
            if response is None:
                raise RuntimeError("All fallback models failed.")

            msg = response.choices[0].message
            tool_calls = None
            if getattr(msg, "tool_calls", None):
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments),
                    }
                    for tc in msg.tool_calls
                ]
            content = msg.content or ""
            usage = {
                "input_tokens": (
                    int(response.usage.prompt_tokens) if response.usage else 0
                ),
                "output_tokens": (
                    int(response.usage.completion_tokens) if response.usage else 0
                ),
            }
            raw_message = msg

        lf.update_current_generation(
            output=content,
            usage_details={
                "input": int(usage.get("input_tokens", 0)),
                "output": int(usage.get("output_tokens", 0)),
            },
        )

        return {
            "content": content,
            "tool_calls": tool_calls,
            "raw_message": raw_message,
        }

    def _to_langchain_messages(self, messages: list[dict]):
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        converted = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "user":
                converted.append(HumanMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            elif role == "tool":
                converted.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=message.get("tool_call_id", ""),
                    )
                )
        return converted
