"""Two-GPU standard-EP evidence for the Qwen3-MoE multi-LoRA sidecar."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.model.qwen3_moe.lite import multi_lora
from megatron.lite.model.qwen3_moe.lite.model import MoELayer
from megatron.lite.primitive.modules.multi_lora_bank import DenseLoraBank
from megatron.lite.primitive.parallel.state import ParallelState

pytestmark = [
    pytest.mark.mlite,
    pytest.mark.smoke,
    pytest.mark.gpu,
    pytest.mark.distributed,
]


@pytest.fixture(scope="module", autouse=True)
def _ep2_cuda_group():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the EP2 sidecar smoke.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("run this smoke through torchrun with exactly two ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created = False
    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")
        created = True
    yield
    if created:
        dist.destroy_process_group()


def _config():
    return Qwen3MoEConfig(
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        max_position_embeddings=16,
        layer_types=["full_attention"],
    )


def _parallel_state(ep_size: int) -> ParallelState:
    return ParallelState(
        ep_group=dist.group.WORLD if ep_size > 1 else None,
        ep_size=ep_size,
        ep_rank=dist.get_rank() if ep_size > 1 else 0,
        # tp_size=1 must not carry a world-sized tp_group.
        tp_group=None,
        tp_size=1,
        etp_size=1,
    )


def _fixed_remote_router(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Every top-2 route is owned by EP rank 1; rank 0 receives zero tokens."""
    scores = torch.tensor([[0.25, 0.75]], dtype=x.dtype, device=x.device).expand(
        x.shape[0], -1
    )
    indices = torch.tensor([[2, 3]], dtype=torch.long, device=x.device).expand(
        x.shape[0], -1
    )
    return scores, indices


def _bank(device: torch.device):
    # fc1 delta has grouped-GEMM1's 2 * intermediate width; fc2 delta has hidden width.
    return multi_lora.MoELoraSidecar(
        DenseLoraBank(
            torch.randn(2, 2, 16, device=device, dtype=torch.bfloat16),
            torch.randn(2, 16, 2, device=device, dtype=torch.bfloat16),
        ),
        DenseLoraBank(
            torch.randn(2, 2, 8, device=device, dtype=torch.bfloat16),
            torch.randn(2, 16, 2, device=device, dtype=torch.bfloat16),
        ),
        torch.tensor([1, 0], device=device, dtype=torch.long),
        1.0,
    )


def _clone_sidecar(sidecar, *, slots=None):
    return multi_lora.MoELoraSidecar(
        DenseLoraBank(
            sidecar.fc1.a_bank.detach().clone().requires_grad_(True),
            sidecar.fc1.b_bank.detach().clone().requires_grad_(True),
        ),
        DenseLoraBank(
            sidecar.fc2.a_bank.detach().clone().requires_grad_(True),
            sidecar.fc2.b_bank.detach().clone().requires_grad_(True),
        ),
        sidecar.lora_indices if slots is None else slots,
        sidecar.scale,
    )


def _layer(*, ep_size: int, recompute: bool) -> MoELayer:
    layer = MoELayer(
        _config(),
        _parallel_state(ep_size),
        use_deepep=False,
        moe_act_recompute=recompute,
    ).cuda()
    layer.router.forward = _fixed_remote_router
    return layer


@pytest.mark.timeout(120)
def test_standard_ep2_real_te_layer_recompute_and_empty_receive_match_oracle():
    """Exercise actual TokenDispatcher and untouched TE GroupedLinear GEMM1/GEMM2."""
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(1701)
    plain = _layer(ep_size=2, recompute=False)
    recomputed = _layer(ep_size=2, recompute=True)
    recomputed.load_state_dict(plain.state_dict())
    reference = _layer(ep_size=1, recompute=False)
    # Build an independent centralized EP1 oracle.  The exercised routes are
    # global experts 2/3, which are rank 1's local GroupedLinear slots 0/1.
    reference_state = reference.state_dict()
    local_state = plain.state_dict()
    for module in ("experts.fc1", "experts.fc2"):
        for local_slot, global_expert in (
            ("weight0", "weight2"),
            ("weight1", "weight3"),
        ):
            source, target = f"{module}.{local_slot}", f"{module}.{global_expert}"
            assert source in local_state and target in reference_state
            rank1_source = local_state[source].detach().clone()
            dist.broadcast(rank1_source, src=1)
            # The rank-1 owner is the only legitimate source for global 2/3;
            # this catches accidental use of rank 0's local experts 0/1.
            if dist.get_rank() == 1:
                torch.testing.assert_close(
                    rank1_source, local_state[source], rtol=0, atol=0
                )
            reference_state[target].copy_(rank1_source)
            torch.testing.assert_close(
                reference_state[target], rank1_source, rtol=0, atol=0
            )
    reference.load_state_dict(reference_state)
    # Local inputs differ, whereas replicated LoRA banks must agree before EP grad sync.
    torch.manual_seed(1701 + dist.get_rank())
    source = _bank(device)
    for tensor in (
        source.fc1.a_bank,
        source.fc1.b_bank,
        source.fc2.a_bank,
        source.fc2.b_bank,
    ):
        dist.broadcast(tensor, src=0)
    plain_sidecar, recomputed_sidecar = _clone_sidecar(source), _clone_sidecar(source)
    gathered_x = [
        torch.empty(2, 16, device=device, dtype=torch.bfloat16) for _ in range(2)
    ]
    x_plain = torch.randn(
        2, 16, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    dist.all_gather(gathered_x, x_plain.detach())
    x_oracle = torch.cat(gathered_x).detach().requires_grad_(True)
    oracle_sidecar = _clone_sidecar(
        source, slots=torch.tensor([1, 0, 1, 0], device=device)
    )
    x_recomputed = x_plain.detach().clone().requires_grad_(True)

    received = []
    original_forward = plain.experts.forward

    def capture_forward(*args, **kwargs):
        received.append(args[1].detach().clone())
        return original_forward(*args, **kwargs)

    plain.experts.forward = capture_forward

    # This EP1 call is independent of dispatch-sidecar: it evaluates the same
    # global tokens, slots, and rank-1 TE expert weights without any AllToAll.
    oracle_out = reference(x_oracle, multi_lora_sidecar=oracle_sidecar)
    oracle_out.float().sum().backward()
    plain_out = plain(x_plain, multi_lora_sidecar=plain_sidecar)
    plain_out.float().sum().backward()
    recomputed_out = recomputed(x_recomputed, multi_lora_sidecar=recomputed_sidecar)
    recomputed_out.float().sum().backward()

    assert received and int(received[0].sum()) == (0 if dist.get_rank() == 0 else 8)
    start = dist.get_rank() * x_plain.shape[0]
    for actual_out, actual_x, actual_sidecar in (
        (plain_out, x_plain, plain_sidecar),
        (recomputed_out, x_recomputed, recomputed_sidecar),
    ):
        torch.testing.assert_close(
            actual_out, oracle_out[start : start + 2], rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            actual_x.grad, x_oracle.grad[start : start + 2], rtol=2e-2, atol=2e-2
        )
        for actual, oracle in (
            (actual_sidecar.fc1.a_bank.grad, oracle_sidecar.fc1.a_bank.grad),
            (actual_sidecar.fc1.b_bank.grad, oracle_sidecar.fc1.b_bank.grad),
            (actual_sidecar.fc2.a_bank.grad, oracle_sidecar.fc2.a_bank.grad),
            (actual_sidecar.fc2.b_bank.grad, oracle_sidecar.fc2.b_bank.grad),
        ):
            assert actual is not None
            torch.testing.assert_close(actual, oracle, rtol=2e-2, atol=2e-2)
            gathered = [torch.empty_like(actual) for _ in range(2)]
            dist.all_gather(gathered, actual)
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("etp_size", "use_deepep", "message"), [(2, False, "ETP"), (1, True, "DeepEP")]
)
def test_sidecar_rejects_etp_and_deepep_before_dispatch(etp_size, use_deepep, message):
    layer = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    layer.ps = SimpleNamespace(etp_size=etp_size)
    layer._use_deepep_requested = use_deepep
    bank = DenseLoraBank(
        torch.ones(1, 1, 2, device="cuda"), torch.ones(1, 2, 1, device="cuda")
    )
    sidecar = multi_lora.MoELoraSidecar(
        bank, bank, torch.zeros(1, device="cuda", dtype=torch.long), 1.0
    )
    with pytest.raises(RuntimeError, match=message):
        layer(torch.ones(1, 2, device="cuda"), multi_lora_sidecar=sidecar)
