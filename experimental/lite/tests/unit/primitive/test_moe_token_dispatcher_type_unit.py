# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Backend selection for moe_token_dispatcher_type.

These run on CPU with neither DeepEP nor HybridEP installed, which is exactly the state
that used to be indistinguishable from "the backend is off": the old `use_deepep` flag
was ANDed with `deep_ep is not None`, so a requested DeepEP run silently became an
AllToAll run. The typed option is required to say so instead.
"""

import pytest
import torch

from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.parallel import ParallelState


def _ps(ep_size: int) -> ParallelState:
    return ParallelState(ep_size=ep_size, ep_rank=0)


def _build(ep_size: int, dispatcher_type: str) -> TokenDispatcher:
    return TokenDispatcher(
        num_experts=4,
        hidden_size=2,
        ps=_ps(ep_size),
        moe_token_dispatcher_type=dispatcher_type,
    )


@pytest.mark.parametrize(
    "dispatcher_type, hint",
    [("deepep", "github.com/deepseek-ai/DeepEP"), ("hybridep", "hybrid-ep")],
)
def test_requesting_an_uninstalled_backend_fails_loud(dispatcher_type, hint):
    # Neither package is importable here, so this is the "dependency missing" case.
    with pytest.raises(RuntimeError) as excinfo:
        _build(ep_size=2, dispatcher_type=dispatcher_type)

    message = str(excinfo.value)
    assert dispatcher_type in message
    assert "not installed" in message
    # The error has to be actionable: where to get it, and how to run without it.
    assert hint in message
    assert "alltoall" in message


@pytest.mark.parametrize("dispatcher_type", ["deepep", "hybridep"])
def test_uninstalled_backend_does_not_silently_become_alltoall(dispatcher_type):
    # The regression guard proper. Under the old `use_deepep` flag this construction
    # succeeded and quietly ran AllToAll; nothing downstream could tell.
    with pytest.raises(RuntimeError):
        _build(ep_size=2, dispatcher_type=dispatcher_type)


def test_alltoall_needs_no_optional_dependency():
    dispatcher = _build(ep_size=2, dispatcher_type="alltoall")
    assert dispatcher.backend == "alltoall"


@pytest.mark.parametrize("dispatcher_type", ["alltoall", "deepep", "hybridep"])
def test_single_ep_rank_degenerates_to_local_for_every_backend(dispatcher_type):
    # With one expert-parallel rank there is no dispatch to perform, so no backend is
    # required to be installed -- including the ones that are missing here.
    dispatcher = _build(ep_size=1, dispatcher_type=dispatcher_type)
    assert dispatcher.backend == "local"
    assert dispatcher.moe_token_dispatcher_type == dispatcher_type


@pytest.mark.parametrize("dispatcher_type", ["allgather", "flex", "DeepEP", "", None, True])
def test_unknown_dispatcher_type_is_rejected(dispatcher_type):
    with pytest.raises(ValueError, match="Unknown moe_token_dispatcher_type"):
        _build(ep_size=2, dispatcher_type=dispatcher_type)


def test_alltoall_and_deepep_share_the_local_roundtrip_at_ep1():
    # Guards the AC-3 claim that the rename left the data path alone: the two types are
    # bit-identical wherever they run the same code.
    hidden = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    topk_indices = torch.tensor([[0], [2], [1], [2]])
    topk_scores = torch.ones(4, 1)

    outputs = []
    for dispatcher_type in ("alltoall", "deepep"):
        dispatcher = _build(ep_size=1, dispatcher_type=dispatcher_type)
        dispatched, tokens_per_expert, probs = dispatcher.dispatch(
            hidden, topk_scores, topk_indices
        )
        outputs.append((dispatched, tokens_per_expert, probs, dispatcher.combine(dispatched)))

    for left, right in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
