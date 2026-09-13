import json

import pytest
import torch
from megatron.lite.model.qwen3_8_flash_next.engram import (
    Qwen3_8_FlashNextEngramTableConfig,
)
from megatron.lite.model.registry import resolve_runtime_model_name


def test_training_runtime_is_registered():
    assert (
        resolve_runtime_model_name('qwen3_8_flash_next', 'lite') == 'qwen3_8_flash_next'
    )


def test_protocol_preserves_wrapped_config_metadata():
    from megatron.lite.model.qwen3_8_flash_next.protocol import build_model_config

    source = {
        'model_type': 'qwen4_exp',
        'text_config': {},
        'vision_config': {'depth': 27},
    }
    config = build_model_config(source, num_hidden_layers=2)
    assert config.num_hidden_layers == 2 and config.vision_config == {'depth': 27}
    assert source['text_config'] == {}


def test_owner_table_lookup_accumulates_duplicate_gradients():
    table = Qwen3_8_FlashNextEngramTableConfig(4, 2).build(
        process_group=None, device='cpu', dtype=torch.float32
    )
    with torch.no_grad():
        table.weight.copy_(torch.arange(8).reshape(4, 2))
    output = table(torch.tensor([[3, 1, 3]]))
    torch.testing.assert_close(
        output, torch.tensor([[[6.0, 7.0], [2.0, 3.0], [6.0, 7.0]]])
    )
    output.sum().backward()
    torch.testing.assert_close(
        table.weight.grad,
        torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0]]),
    )


def test_protocol_next_token_targets_stop_at_document_boundaries():
    from megatron.lite.model.qwen3_8_flash_next.protocol import _forward_step
    from megatron.lite.runtime.contracts import PackedBatch

    ids = torch.tensor([1, 2, 3, 4, 5])
    batch = PackedBatch(
        ids, ids, torch.tensor([3, 2]), loss_mask=torch.tensor([1, 1, 0, 1, 1])
    )
    captured = _forward_step(lambda **kwargs: kwargs, batch)
    assert captured['labels'].tolist() == [[2, -100, -100, 5, -100]]


def tiny_training_config():
    return dict(
        model_type='qwen4_exp_text',
        hidden_size=128,
        num_hidden_layers=2,
        vocab_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        hc_count=4,
        hc_lowrank=16,
        ple_layer_ids=[2],
        ple_embed_dim=128,
        layer_types=['linear_attention', 'full_attention'],
        eos_token_id=127,
        indexer_n_heads=2,
        indexer_head_dim=128,
        indexer_compress_ratio=4,
        indexer_budget=16,
    )


def test_model_allows_data_parallel_replicas(
    transformer_engine_import_stub, monkeypatch
):
    transformer_engine_import_stub()
    from types import SimpleNamespace

    from megatron.lite.model.qwen3_8_flash_next import model as module
    from megatron.lite.model.qwen3_8_flash_next.protocol import build_model_config

    monkeypatch.setattr(module, 'Qwen38Layer', lambda *a, **kw: torch.nn.Identity())
    ps = SimpleNamespace(
        tp_size=1, ep_size=1, etp_size=1, cp_size=1, pp_size=1, dp_size=2
    )
    model = module.Qwen38Model(build_model_config(tiny_training_config()), ps)
    assert model.ps.dp_size == 2


@pytest.mark.parametrize('ep', [1, 2])
def test_model_allows_expert_owners(transformer_engine_import_stub, monkeypatch, ep):
    transformer_engine_import_stub()
    from types import SimpleNamespace

    from megatron.lite.model.qwen3_8_flash_next import model as module
    from megatron.lite.model.qwen3_8_flash_next.protocol import build_model_config

    monkeypatch.setattr(module, 'Qwen38Layer', lambda *a, **kw: torch.nn.Identity())
    ps = SimpleNamespace(
        tp_size=1, ep_size=ep, etp_size=1, cp_size=1, pp_size=1, dp_size=2
    )
    assert (
        module.Qwen38Model(build_model_config(tiny_training_config()), ps).ps.ep_size
        == ep
    )


def test_protocol_expert_checkpoint_placement():
    from megatron.lite.model.qwen3_8_flash_next.protocol import parameter_placements
    from torch.distributed.tensor import Replicate, Shard

    for suffix in ('fc1.weight0', 'fc2.weight1'):
        places = parameter_placements('layers.0.mlp.experts.' + suffix)
        assert places == [Replicate(), Replicate(), Shard(0), Replicate()]
    for name in (
        'layers.0.mlp.router.gate.weight',
        'layers.0.mlp.shared_expert.shared_gate.weight',
        'layers.1.ple.embedding.table.weight',
    ):
        assert parameter_placements(name) == [Replicate()] * 4


@pytest.fixture
def isolated_training_groups():
    from megatron.core import parallel_state as mpu

    was_initialized = torch.distributed.is_initialized()
    had_model_parallel = mpu.is_initialized()
    try:
        yield
    finally:
        if not had_model_parallel:
            mpu.destroy_model_parallel()
        if not was_initialized and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.gpu
def test_native_runtime_trains_all_decoder_branches(
    tmp_path, monkeypatch, isolated_training_groups
):
    from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
    from megatron.lite.primitive.modules import gated_delta_net as gdn_primitive
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
    from megatron.lite.runtime.contracts import OptimizerConfig, PackedBatch

    torch.manual_seed(1234)
    backends = {}
    for name in ('torch_chunk_gated_delta_rule', '_fla_chunk_gated_delta_rule'):
        if not hasattr(gdn_primitive, name):
            continue
        original = getattr(gdn_primitive, name)

        def counted(*args, original=original, name=name, **kwargs):
            backends[name] = backends.get(name, 0) + 1
            return original(*args, **kwargs)

        monkeypatch.setattr(gdn_primitive, name, counted)

    config = tiny_training_config()
    (tmp_path / 'config.json').write_text(json.dumps(config))
    cfg = MegatronLiteConfig(
        model_name='qwen3_8_flash_next',
        hf_path=str(tmp_path),
        load_hf_weights=False,
        optimizer=OptimizerConfig(lr=0.003),
        impl_cfg={
            'ngram_primes': (
                17,
                19,
                23,
                29,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
                67,
                71,
                73,
                79,
            )
        },
    )
    runtime = MegatronLiteRuntime(str(tmp_path), cfg)
    handle = runtime.build_model()
    model = unwrap_model(handle._model)
    gdn = model.layers[0].linear_attn
    x = torch.linspace(-3, 4, 2 * 128, device='cuda', dtype=torch.bfloat16).reshape(
        1, 2, 128
    )
    gate = torch.linspace(-2, 1, x.numel(), device='cuda', dtype=x.dtype).reshape_as(x)
    expected_norm = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + config.get('rms_norm_eps', 1e-6)
    )
    expected_norm = (
        expected_norm * gdn.norm.weight.float() * gate.float().sigmoid()
    ).to(x.dtype)
    torch.testing.assert_close(
        gdn._apply_gated_norm(x, gate).reshape_as(x), expected_norm, rtol=0, atol=0
    )
    projection = gdn.in_proj(x.transpose(0, 1)).transpose(0, 1)
    torch.testing.assert_close(
        projection,
        torch.nn.functional.linear(x, gdn.in_proj.linear.weight),
        rtol=0.01,
        atol=0.01,
    )
    required = [
        'layers.0.linear_attn.in_proj',
        'layers.0.linear_attn.norm',
        'layers.1.self_attn.q_proj',
        'hyper_connection_mixer',
        'layers.1.ple.ple_embedding',
        'layers.1.ple.conv1d',
    ] + [
        f'layers.{i}.{branch}'
        for i in range(2)
        for branch in (
            'attn_hyper_connection',
            'mlp_hyper_connection',
            'mlp.experts',
            'mlp.router',
            'mlp.shared_expert',
        )
    ]
    gradients = set()
    hooks = []
    for name, param in model.named_parameters():
        if param.requires_grad:

            def capture(grad, name=name):
                assert torch.isfinite(grad).all(), name
                if grad.float().norm() > 0:
                    gradients.update(key for key in required if key in name)

            hooks.append(param.register_hook(capture))
    before = model.layers[1].ple.ple_embedding.ngram_embedding.weight.detach().clone()
    ids = torch.arange(64, device='cuda') % 32
    batch = PackedBatch(ids, ids.clone(), torch.tensor([64], device='cuda'))
    losses = []

    def loss_fn(out, batch):
        if not losses:
            pending, visited, leaves = [out['loss'].grad_fn], set(), set()
            while pending:
                node = pending.pop()
                if node is None or node in visited:
                    continue
                visited.add(node)
                if hasattr(node, 'variable'):
                    leaves.add(id(node.variable))
                pending.extend(edge[0] for edge in node.next_functions)
            missing = [
                name
                for name, param in model.named_parameters()
                if param.requires_grad and id(param) not in leaves
            ]
            assert not missing, ('TRAINING_GRAPH_MISSING_PARAMETERS', missing)
        losses.append(float(out['loss'].detach()))
        return out['loss'], {}

    try:
        for _ in range(12):
            runtime.zero_grad(handle)
            runtime.forward_backward(handle, [batch], loss_fn)
            success, norm, _ = runtime.optimizer_step(handle)
            assert success and 0 < norm < float('inf')
        assert all(torch.isfinite(torch.tensor(losses)))
        assert losses[-1] < losses[0] * 0.9, losses
        assert gradients == set(required), sorted(set(required) - gradients)
        assert not torch.equal(
            before, model.layers[1].ple.ple_embedding.ngram_embedding.weight
        )
        assert all(
            p.grad is None for p in model.layers[1].self_attn.indexer.parameters()
        )
        torch.save(model.state_dict(), tmp_path / 'model.pt')
        expected = model.lm_head.weight.detach().clone()
        with torch.no_grad():
            model.lm_head.weight.zero_()
        model.load_state_dict(torch.load(tmp_path / 'model.pt', weights_only=True))
        assert torch.equal(expected, model.lm_head.weight)
        print(
            'QWEN38_STAGE1_LOSSES',
            losses,
            'GRADIENT_BRANCHES',
            sorted(gradients),
            'GDN_BACKENDS',
            backends,
        )
    finally:
        for hook in hooks:
            hook.remove()
