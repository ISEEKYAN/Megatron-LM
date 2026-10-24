# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 multi-turn codec: incremental tokens equal a full V4.1 render; tool calls round-trip.

Guards the train/rollout token contract of the SWE agent loop: the sequence the
rollout builds turn by turn (initial render + generated assistant tokens +
appended tool turns) must be exactly what ``encode_messages`` renders for the
whole conversation, otherwise trained tokens differ from what was sampled.
Needs the DS4.1 tokenizer (``DS41_TOKENIZER_PATH``) and vLLM with V4.1 encoding.
"""

import json
import os

import pytest

pytest.importorskip("vllm.tokenizers.deepseek_v41_encoding")
transformers = pytest.importorskip("transformers")

from verl_mlite.agent.ds41_codec import DS41Codec  # noqa: E402

TOKENIZER = os.environ.get("DS41_TOKENIZER_PATH", "")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "stateful_shell",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "submit", "description": "Finish.", "parameters": {"type": "object", "properties": {}}},
    },
]


def _assistant(name, args, content=""):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"id": "c0", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}],
    }


@pytest.fixture(scope="module")
def tokenizer():
    if not os.path.isdir(TOKENIZER):
        pytest.skip("DS4.1 tokenizer not available")
    return transformers.AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.mark.parametrize("thinking", [False, True])
def test_incremental_equals_full_render(tokenizer, thinking):
    codec = DS41Codec(tokenizer, thinking=thinking)
    turns = [
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Fix the bug."}],
        [_assistant("stateful_shell", {"command": "ls /app", "timeout": 30}, content="Look first.")],
        [{"role": "tool", "tool_call_id": "c0", "name": "stateful_shell", "content": "a.py\nb.py"}],
        [_assistant("stateful_shell", {"command": "cat a.py"})],
        [{"role": "tool", "tool_call_id": "c0", "name": "stateful_shell", "content": "print(1)"}],
        [_assistant("submit", {})],
    ]
    committed = list(turns[0])
    ids = codec.initial(committed, TOOLS)
    for turn in turns[1:]:
        if turn[0]["role"] == "assistant":
            before = codec.full(committed, TOOLS)
            after = codec.full(committed + turn, TOOLS)
            assert after[: len(before)] == before
            generated = after[len(before) :]
            if thinking:
                # A sampled turn carries its reasoning; drop_thinking keeps it with tools present.
                assert generated[-1] == codec.eos_id
            ids += generated
            content, calls = codec.parse(generated, TOOLS)
            assert [c["function"]["name"] for c in calls] == [turn[0]["tool_calls"][0]["function"]["name"]]
            assert json.loads(calls[0]["function"]["arguments"]) == json.loads(
                turn[0]["tool_calls"][0]["function"]["arguments"]
            )
        else:
            ids += codec.append(committed, turn, TOOLS)
        committed += turn
    assert ids == codec.full(committed, TOOLS)


def test_plain_text_turn_has_no_tool_calls(tokenizer):
    codec = DS41Codec(tokenizer, thinking=False)
    ids = tokenizer.encode("All done.", add_special_tokens=False) + [codec.eos_id]
    content, calls = codec.parse(ids, TOOLS)
    assert calls == [] and content.strip() == "All done."
