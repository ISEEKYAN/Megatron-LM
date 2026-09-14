"""Whole-model CP wiring and distinct real-token/loss populations."""

from types import SimpleNamespace

import torch
from torch import nn


def test_cp_router_counts_real_tokens_and_scales_aux(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import router as module

    config = SimpleNamespace(
        num_experts_per_tok=2,
        num_experts=4,
        hidden_size=128,
        router_aux_loss_coef=0.001,
    )
    router = module.TopKRouter(
        config, SimpleNamespace(tp_size=1), router_dtype=torch.float32
    )
    mask = torch.tensor([True] * 5 + [False] * 3)
    captured = {}
    original = module.switch_load_balancing_loss_func

    def loss(probs, counts, total, *args, **kwargs):
        captured.update(
            probs=probs.detach().clone(), counts=counts.clone(), total=total
        )
        return original(probs, counts, total, *args, **kwargs)

    def reduce(values, group):
        captured['local'] = values.clone()
        values.add_(torch.tensor([4, 4, 4, 4, 8]))

    monkeypatch.setattr(module, 'switch_load_balancing_loss_func', loss)
    monkeypatch.setattr(module.dist, 'all_reduce', reduce)
    torch.manual_seed(95)
    x = torch.randn(8, 128, requires_grad=True)
    scores, _ = router(x, token_mask=mask, token_group=object(), aux_loss_scale=2)
    (scores.sum() * 0).backward()
    doubled = x.grad.clone()
    x.grad = None
    scores, _ = router(x, token_mask=mask, token_group=object(), aux_loss_scale=1)
    (scores.sum() * 0).backward()
    assert torch.equal(doubled, x.grad * 2), 'CP_ROUTER_AUX_DDP_SCALE'
    assert not doubled[5:].count_nonzero(), 'CP_ROUTER_PADDING_AUX_GRAD_ZERO'
    assert captured['local'][-1] == 5, 'CP_ROUTER_LOCAL_REAL_TOKEN_COUNT'
    assert captured['local'][:-1].sum() == 10, 'CP_ROUTER_MASKED_EXPERT_COUNT'
    assert captured['total'] == 13, 'CP_ROUTER_GLOBAL_REAL_TOKEN_COUNT'
    assert not captured['probs'][5:].count_nonzero(), 'CP_ROUTER_PADDING_AUX_ZERO'


def test_cp_layer_passes_context_to_components(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_8_flash_next.model import Qwen38Layer

    class Capture(nn.Module):
        def forward(self, x, *args, **kw):
            self.kw = kw
            return x

    class Mix(nn.Module):
        def mix(self, x):
            return x, x

        def combine(self, x, residual):
            return x + residual

    layer = Qwen38Layer.__new__(Qwen38Layer)
    nn.Module.__init__(layer)
    layer.ple, layer.self_attn, layer.mlp = Capture(), Capture(), Capture()
    layer.linear_attn = None
    layer.attn_hyper_connection, layer.mlp_hyper_connection = Mix(), Mix()
    context = SimpleNamespace(
        global_padding_mask=torch.tensor([[False] * 13 + [True] * 3]),
        local_sequence_start=8,
        local_sequence_end=16,
        group=object(),
        size=2,
    )
    layer(
        torch.zeros(1, 8, 128),
        torch.zeros(1, 8, dtype=torch.long),
        torch.zeros(1, 8, 1, 64),
        torch.tensor([0, 5, 13, 16]),
        cp_context=context,
    )
    assert layer.ple.kw['cp_context'] is context, 'CP_LAYER_PLE_CONTEXT'
    assert layer.self_attn.kw['cp_context'] is context, 'CP_LAYER_QSA_CONTEXT'
    assert (
        layer.self_attn.kw.get('cu_seqlens') is None
    ), 'CP_LAYER_REAL_DOCUMENT_CONTEXT'
    assert (
        layer.mlp.kw['token_mask'].tolist() == [True] * 5 + [False] * 3
    ), 'CP_LAYER_ROUTER_REAL_MASK'
    assert layer.mlp.kw['aux_loss_scale'] == 2, 'CP_LAYER_AUX_DDP_COMPENSATION'


def test_cp_model_uses_global_loss_population(
    transformer_engine_import_stub, monkeypatch
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_8_flash_next import model as module
    from megatron.lite.model.qwen3_8_flash_next import protocol
    from megatron.lite.runtime.contracts import PackedBatch
    from test_qwen38_training import tiny_training_config

    class Layer(nn.Module):
        def forward(self, hidden, *args, **kwargs):
            self.context = kwargs.get('cp_context')
            return hidden

    monkeypatch.setattr(module, 'Qwen38Layer', lambda *a, **kw: Layer())
    ps = SimpleNamespace(
        tp_size=1,
        ep_size=1,
        etp_size=1,
        cp_size=2,
        pp_size=1,
        dp_size=1,
        cp_rank=1,
        cp_group=object(),
    )
    model = module.Qwen38Model(protocol.build_model_config(tiny_training_config()), ps)
    ids = torch.arange(13)
    mask = torch.ones(13, dtype=torch.bool)
    mask[10] = False
    result = protocol._forward_step(
        model, PackedBatch(ids, ids.clone(), torch.tensor([5, 8]), loss_mask=mask)
    )
    expected = (
        torch.nn.functional.cross_entropy(
            result['logits'].float().flatten(0, 1),
            torch.tensor([9, -100, 11, 12, -100, -100, -100, -100]),
            reduction='none',
        ).sum()
        / 10
        * 2
    )
    assert torch.equal(result['loss'], expected), 'CP_MODEL_GLOBAL_LOSS_DDP_SCALE'
    assert model.layers[0].context.size == 2, 'CP_MODEL_LAYER_CONTEXT'


def test_cp_reference_moe_reuses_one_view(monkeypatch, transformer_engine_import_stub):
    transformer_engine_import_stub()
    import importlib.util
    from contextlib import nullcontext
    from pathlib import Path

    from megatron.lite.model.qwen3_5.lite.model import SharedExpert
    from megatron.lite.primitive.utils import moe as utils

    path = Path(__file__).parents[2] / 'smoke/workflows/training/qwen38_cp_reference.py'
    spec = importlib.util.spec_from_file_location('cp_reference_under_test', path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    stream = SimpleNamespace(wait_stream=lambda _: None)
    monkeypatch.setattr(SharedExpert, '_get_stream', lambda: stream)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: stream)
    monkeypatch.setattr(torch.cuda, 'stream', lambda _: nullcontext())
    seen = {}
    events = []

    class Shared(nn.Module):
        def __init__(self, rank):
            super().__init__()
            self.rank = rank

        def forward(self, x):
            seen[self.rank] = x
            events.append(('shared', self.rank))
            return x * 0.1

    class Dispatch:
        def __init__(self, rank):
            self.rank = rank

        def dispatch(self, x, scores, indices):
            assert x is seen[self.rank], 'CP_REFERENCE_COMMON_MOE_VIEW'
            return x, None, scores

        def wait_dispatch_event(self):
            pass

        def combine(self, x):
            return x

    class Expert(nn.Module):
        def forward(self, x, *args, **kwargs):
            return x * 0.3

    modules = []
    for rank in range(2):
        r = SimpleNamespace(
            gate=nn.Linear(128, 4, bias=False),
            router_dtype=torch.float32,
            topk=2,
            use_pre_softmax=False,
            num_experts=4,
            aux_loss_coeff=0.001,
        )
        modules.append(
            SimpleNamespace(
                router=r,
                shared_expert=Shared(rank),
                dispatcher=Dispatch(rank),
                experts=Expert(),
                preserve_3d_graph=False,
            )
        )

    def gate(x, weight, bias, dtype):
        rank = next(i for i, m in enumerate(modules) if m.router.gate.weight is weight)
        assert (
            rank in seen and x is seen[rank]
        ), 'CP_REFERENCE_SHARED_BEFORE_ROUTER_SAME_VIEW'
        events.append(('gate', rank))
        return F.linear(x.float(), weight.float())

    from torch.nn import functional as F

    monkeypatch.setattr(utils, 'router_gating_linear', gate)
    xs = [torch.randn(1, 8, 128, requires_grad=True) for _ in range(2)]
    outputs = ref.moe(modules, xs, torch.tensor([[True] * 13 + [False] * 3]))
    assert len(outputs) == 2 and events == [
        ('shared', 0),
        ('gate', 0),
        ('shared', 1),
        ('gate', 1),
    ]
