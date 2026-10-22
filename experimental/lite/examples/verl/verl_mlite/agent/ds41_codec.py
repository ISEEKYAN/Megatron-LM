# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Token-in/token-out multi-turn codec for DeepSeek-V4.1 tool calling.

The release ships no jinja template; vLLM encodes V4.1 prompts with
``vllm.tokenizers.deepseek_v41_encoding.encode_messages`` and parses tool calls
with the ``deepseek_v41`` parser. This codec uses the same two pieces so the
rollout prompt is byte-identical to what ``vllm serve`` would build, while
keeping every generated token as sampled (no re-tokenization of assistant
turns): appended tool/user turns are rendered with the earlier messages as
``context`` and concatenated after the generated tokens.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

EOS_TOKEN = "<｜end▁of▁sentence｜>"


def _encoding():
    from vllm.tokenizers import deepseek_v41_encoding

    return deepseek_v41_encoding


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "\n\n".join(block.get("text", "") for block in content)
    return content or ""


class DS41Codec:
    """Render V4.1 prompts incrementally and parse V4.1 tool calls."""

    def __init__(self, tokenizer: Any, *, thinking: bool, reasoning_effort: Any = None):
        self.tokenizer = tokenizer
        self.thinking_mode = "thinking" if thinking else "chat"
        self.reasoning_effort = reasoning_effort
        self.eos_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        self._parser = None
        self._parser_tools = None

    def _conversation(self, messages: list[dict[str, Any]], tools: list[dict] | None) -> list[dict[str, Any]]:
        conv = []
        for message in copy.deepcopy(messages):
            out = {"role": message["role"], "content": _text(message.get("content"))}
            if message.get("tool_calls"):
                out["tool_calls"] = message["tool_calls"]
            if message.get("reasoning_content"):
                out["reasoning_content"] = message["reasoning_content"]
            if message["role"] == "tool":
                out["tool_call_id"] = message.get("tool_call_id")
            conv.append(out)
        if tools:
            system = next((m for m in conv if m["role"] == "system"), None)
            if system is None:
                system = {"role": "system", "content": ""}
                conv.insert(0, system)
            system["tools"] = tools
        return conv

    def _encode_text(self, messages, context=None) -> str:
        return _encoding().encode_messages(
            messages,
            thinking_mode=self.thinking_mode,
            context=context,
            drop_thinking=True,
            reasoning_effort=self.reasoning_effort,
        )

    def _ids(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def initial(self, messages: list[dict[str, Any]], tools: list[dict] | None) -> list[int]:
        return self._ids(self._encode_text(self._conversation(messages, tools)))

    def append(
        self, committed: list[dict[str, Any]], appended: list[dict[str, Any]], tools: list[dict] | None
    ) -> list[int]:
        """Tokens for ``appended`` (tool/user turns) rendered after ``committed``."""
        if any(m["role"] == "assistant" for m in appended):
            raise ValueError("only non-assistant turns can be appended")
        conv = self._conversation(committed + appended, tools)
        cut = len(conv) - len(appended)
        return self._ids(self._encode_text(conv[cut:], context=conv[:cut]))

    def full(self, messages: list[dict[str, Any]], tools: list[dict] | None) -> list[int]:
        """Reference render of a whole conversation (tests only)."""
        return self.initial(messages, tools)

    def parse(self, token_ids: list[int], tools: list[dict]) -> tuple[str, list[dict]]:
        """Split a generated turn into (content, OpenAI tool_calls) with vLLM's V4.1 parser."""
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
        from vllm.tool_parsers import ToolParserManager

        key = json.dumps(tools, sort_keys=True)
        vllm_tools = [ChatCompletionToolsParam(**tool) for tool in tools]
        if self._parser is None or self._parser_tools != key:
            parser_cls = ToolParserManager.get_tool_parser("deepseek_v41")
            try:
                self._parser = parser_cls(self.tokenizer, tools=vllm_tools)
            except TypeError:
                self._parser = parser_cls(self.tokenizer)
            self._parser_tools = key
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        if text.endswith(EOS_TOKEN):
            text = text[: -len(EOS_TOKEN)]
        if self.thinking_mode == "thinking" and "</think>" in text:
            text = text.split("</think>", 1)[1]
        request = SimpleNamespace(tools=vllm_tools, tool_choice="auto", skip_special_tokens=True)
        parsed = self._parser.extract_tool_calls(text, request)
        if not parsed.tools_called:
            return text, []
        calls = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for i, tc in enumerate(parsed.tool_calls)
        ]
        return parsed.content or "", calls
