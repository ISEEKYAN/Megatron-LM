# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

VERL_EXAMPLE_ROOT = Path(__file__).resolve().parents[3] / "examples" / "verl"
if str(VERL_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))

from verl_mlite import compat

pytestmark = pytest.mark.optional


def install_server(monkeypatch):
    calls = []

    class Server:
        async def run_server(self, args):
            calls.append(args)
            return "actual-server-result"

        async def run_headless(self, args):
            calls.append(args)
            return "actual-headless-result"

    module = ModuleType(compat._VLLM_ASYNC_SERVER_MODULE)
    module.vLLMHttpServer = Server
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(compat, "_vllm_importable", lambda: True)
    return Server, calls


def arguments(**overrides):
    values = dict(
        nnodes=2,
        node_rank=0,
        data_parallel_size=8,
        data_parallel_size_local=4,
        data_parallel_start_rank=0,
        data_parallel_hybrid_lb=False,
        data_parallel_external_lb=False,
        data_parallel_master_port=31000,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("nodes", [2, 3])
def test_single_frontend_uses_central_routing_without_changing_rank_layout(
    monkeypatch, nodes
):
    server, calls = install_server(monkeypatch)
    args = arguments(nnodes=nodes, data_parallel_size=4 * nodes)
    before = vars(args).copy()
    assert compat._patch_verl_vllm_multinode_dp()
    assert not compat._patch_verl_vllm_multinode_dp()
    assert asyncio.run(server().run_server(args)) == "actual-server-result"
    normalized = calls[0]
    assert normalized is not args
    assert vars(args) == before
    assert vars(normalized) == dict(before, data_parallel_start_rank=None)
    # Rank zero is still inferred from node_rank, while omitting start_rank
    # avoids Native's inference that every node has a hybrid-LB frontend.
    assert normalized.node_rank * normalized.data_parallel_size_local == 0


@pytest.mark.parametrize(
    "overrides",
    [
        dict(nnodes=1, data_parallel_size=4),
        dict(data_parallel_size=1),
        dict(node_rank=1, data_parallel_start_rank=4),
        dict(data_parallel_start_rank=None),
        dict(data_parallel_hybrid_lb=True),
        dict(data_parallel_external_lb=True),
    ],
)
def test_other_topologies_and_explicit_lb_modes_keep_original_arguments(
    monkeypatch, overrides
):
    server, calls = install_server(monkeypatch)
    args = arguments(**overrides)
    before = vars(args).copy()
    assert compat._patch_verl_vllm_multinode_dp()
    assert asyncio.run(server().run_server(args)) == "actual-server-result"
    assert calls == [args] and calls[0] is args
    assert vars(args) == before


def test_remote_headless_start_rank_and_method_are_untouched(monkeypatch):
    server, calls = install_server(monkeypatch)
    original = server.run_headless
    args = arguments(node_rank=1, data_parallel_start_rank=4)
    before = vars(args).copy()
    assert compat._patch_verl_vllm_multinode_dp()
    assert server.run_headless is original
    assert asyncio.run(server().run_headless(args)) == "actual-headless-result"
    assert calls[0] is args and vars(args) == before


def test_no_vllm_does_not_import_server_module(monkeypatch):
    monkeypatch.setattr(compat, "_vllm_importable", lambda: False)
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        lambda name: pytest.fail("server imported without vLLM"),
    )
    assert not compat._patch_verl_vllm_multinode_dp()
