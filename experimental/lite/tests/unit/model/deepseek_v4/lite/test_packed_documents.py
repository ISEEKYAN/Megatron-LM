# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real V4 assembly on CPU, replacing only TE/CUDA kernel and device boundaries.

CSA projections/RoPE, mHC, hash/learned routers, dispatcher, experts and heads
are production modules. Sparse attention and TE GroupedLinear/RMSNorm use Torch
CPU implementations. This does not claim validation of the CUDA kernels.
"""
import hashlib
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

_CORE_SYMBOLS = {
    "megatron.core.tensor_parallel.mappings": ["gather_from_sequence_parallel_region"],
    "megatron.core.transformer.experimental_attention_variant": [
        "csa_cp_layout_kernels",
        "csa_cp_utils",
    ],
    "megatron.core.transformer.experimental_attention_variant.csa": [
        "_unfused_indexer_sparse_attn_from_topk",
        "unfused_compressed_sparse_attn",
    ],
    "megatron.core.transformer.experimental_attention_variant.csa_kernels": [
        "FusedCSAIndexerSparseAttnFromTopkFunc",
        "csa_sparse_attn",
    ],
    "megatron.core.transformer.experimental_attention_variant.dsa": [
        "DSAIndexerLossAutoScaler",
        "DSAIndexerLossLoggingHelper",
    ],
}
_STUBBED_IMPORTERS = (
    "megatron.lite.primitive.modules.attention",
    "megatron.lite.model.deepseek_v4",
)


@pytest.fixture
def cpu_v4(transformer_engine_import_stub, monkeypatch):
    transformer_engine_import_stub()
    before, stubbed = set(sys.modules), False
    try:
        from megatron.lite.model.deepseek_v4.lite import model, protocol
    except ImportError:
        stubbed = True
        for name, symbols in _CORE_SYMBOLS.items():
            module = types.ModuleType(name)
            for symbol in symbols:
                setattr(module, symbol, object())
            monkeypatch.setitem(sys.modules, name, module)
        for name in set(sys.modules) - before:
            if name.startswith(_STUBBED_IMPORTERS):
                del sys.modules[name]
        from megatron.lite.model.deepseek_v4.lite import model, protocol
    from megatron.lite.primitive import transformer_engine as te

    monkeypatch.setattr(te, "RMSNorm", lambda size, eps: nn.RMSNorm(size, eps=eps))
    monkeypatch.setattr(te, "GroupedLinear", _GroupedLinear)
    monkeypatch.setattr(nn.Module, "cuda", lambda self: self)
    from megatron.lite.primitive.parallel import ParallelState

    monkeypatch.setattr(protocol, "init_parallel", lambda _: ParallelState())
    from megatron.lite.primitive.modules.attention import csa

    monkeypatch.setattr(csa, "_load_dsa_kernels", lambda: _CPUSparseKernels())
    from megatron.lite.primitive.modules.router_replay import RouterReplay

    saved_routers = RouterReplay.global_router_replay_instances[:]
    RouterReplay.clear_global_router_replay_instances()
    yield SimpleNamespace(model=model, protocol=protocol)
    RouterReplay.global_router_replay_instances[:] = saved_routers
    # Modules imported against the stubs must not leak into later tests.
    for name in set(sys.modules) - before if stubbed else ():
        if name.startswith(_STUBBED_IMPORTERS):
            del sys.modules[name]


class _GroupedLinear(nn.Module):
    def __init__(self, count, in_features, out_features, **kwargs):
        super().__init__()
        self.count = count
        for i in range(count):
            weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=kwargs['params_dtype'])
            )
            nn.init.normal_(weight, std=0.02)
            self.register_parameter(f'weight{i}', weight)

    def forward(self, x, splits):
        return torch.cat(
            [
                F.linear(part, getattr(self, f'weight{i}'))
                for i, part in enumerate(x.split(splits))
            ]
        )


class _CPUSparseKernels:
    def build_flat_topk_idxs(self, indices, **kwargs):
        return indices.long(), None

    def dsa_sparse_attn(self, query, kv, sinks, indices, scale):
        outputs = []
        for batch in range(query.shape[1]):
            idx = indices[batch]
            values = kv[:, batch][idx.clamp_min(0)]
            scores = (
                torch.einsum('shd,swd->shw', query[:, batch].float(), values.float())
                * scale
            )
            scores = scores.masked_fill((idx < 0)[:, None], float('-inf'))
            sink = sinks.reshape(1, -1, 1).expand(scores.shape[0], -1, -1)
            probs = torch.cat([scores, sink], -1).softmax(-1)[..., :-1]
            outputs.append(
                torch.einsum('shw,swd->shd', probs, values.float()).to(query.dtype)
            )
        return torch.stack(outputs, 1)


def _config(mtp=False):
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size=16,
        hidden_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=4,
        qk_rope_head_dim=2,
        q_lora_rank=4,
        o_lora_rank=4,
        o_groups=2,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        compress_ratios=[0, 0, 0],
        num_hash_layers=1,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        num_nextn_predict_layers=int(mtp),
    )


def _build(protocol, packed=False, recompute=False, mtp=False):
    torch.manual_seed(123)
    kwargs = dict(
        optimizer=None,
        mtp_enable=mtp,
        mtp_enable_train=mtp,
        attention_backend_override='flash',
        recompute=['full'] if recompute else [],
    )
    if packed:
        kwargs['packed_documents'] = True
    return protocol.build_model(
        _config(mtp), impl_cfg=protocol.ImplConfig(**kwargs)
    ).chunks[0]


def _routers(model):
    from megatron.lite.primitive.modules.router_replay import RouterReplay

    routers = []
    for layer in model.layers.values():
        layer.mlp.gate.router_replay = RouterReplay()
        routers.append(layer.mlp.gate.router_replay)
    return routers


def _equal(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.detach().contiguous().reshape(-1).view(torch.uint8),
        expected.detach().contiguous().reshape(-1).view(torch.uint8),
    )


@pytest.mark.parametrize('recompute', [False, True])
def test_v4_packed_outputs_gradients_routes_match_independent_documents(
    cpu_v4, recompute
):
    from megatron.lite.primitive.modules.router_replay import (
        RouterReplayAction as Action,
    )
    from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
    from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

    reference = _build(cpu_v4.protocol)
    packed = _build(cpu_v4.protocol, packed=True, recompute=recompute)
    actual_routers, ref_routers = _routers(packed), _routers(reference)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    params = PackedSeqParams.from_cu_seqlens(cu, 5)
    # Distinct routes from native hash (all zero) guarantee actual substitution.
    targets = [torch.ones(8, 1, dtype=torch.long), torch.arange(8).reshape(-1, 1) % 2]
    for r, target in zip(actual_routers, targets):
        r.router_replay_action = Action.REPLAY_FORWARD
        r.target_topk_idx = target
    ref_outputs, seen = [], [[], []]
    for begin, end in [(0, 3), (3, 8)]:
        for r, target in zip(ref_routers, targets):
            r.router_replay_action = Action.REPLAY_FORWARD
            r.target_topk_idx = target[begin:end]
        ref_outputs.append(reference(input_ids=ids[:, begin:end], enable_mtp=False))
    expected_logits = torch.cat([o['logits'] for o in ref_outputs], 1)
    expected_hidden = torch.cat([o['hidden_states'] for o in ref_outputs], 0)
    labels = torch.tensor([[2, 3, 0, 5, 6, 7, 8, 0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]])
    # Capture each live selection, including recompute calls.
    for r, bucket in zip(actual_routers, seen):
        select = r.select_indices

        def capture(indices, select=select, bucket=bucket):
            selected = select(indices)
            bucket.append(selected.detach().clone())
            return selected

        r.select_indices = capture
    actual = packed(
        input_ids=ids,
        labels=labels,
        loss_mask=mask,
        temperature=2.0,
        calculate_entropy=True,
        packed_seq_params=params,
    )
    inference = packed(input_ids=ids, packed_seq_params=params)
    _equal(inference['logits'], expected_logits)
    _equal(actual['hidden_states'], expected_hidden)
    token_loss = vocab_parallel_cross_entropy((expected_logits / 2).clone(), labels)
    expected_loss = (token_loss * mask).sum() / mask.sum()
    from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy

    _equal(actual['entropy'], vocab_parallel_entropy(expected_logits / 2))
    _equal(actual['log_probs'], -token_loss)
    _equal(actual['loss'], expected_loss)
    for bucket, target in zip(seen, targets):
        _equal(torch.cat(bucket[:2]), target)
    # Replace ambient targets before autograd: document closures must retain theirs.
    for r in actual_routers:
        r.target_topk_idx = torch.zeros_like(targets[0])
        r.router_replay_action = Action.REPLAY_BACKWARD
    actual['loss'].backward()
    expected_loss.backward()
    for (name, parameter), (ref_name, ref_parameter) in zip(
        packed.named_parameters(), reference.named_parameters()
    ):
        assert name == ref_name
        assert (parameter.grad is None) == (ref_parameter.grad is None), name
        if parameter.grad is not None:
            _equal(parameter.grad, ref_parameter.grad)
    if recompute:
        assert all(len(bucket) > 4 for bucket in seen)


_MAIN = 'd8e010069fef5c6da3690d008b899d480864a8de'
_PROTOCOL = 'megatron.lite.model.deepseek_v4.lite.protocol'


def test_default_build_is_byte_identical_to_pinned_main_protocol(cpu_v4, monkeypatch):
    path = 'experimental/lite/' + _PROTOCOL.replace('.', '/') + '.py'
    source = subprocess.check_output(['git', 'show', f'{_MAIN}:{path}'])
    assert (
        hashlib.sha1(b'blob ' + str(len(source)).encode() + b'\0' + source).hexdigest()
        == '5d0441ad00e0839e1f07720c8b44461e1f0e2295'
    )
    spec = importlib.util.spec_from_loader(_PROTOCOL, loader=None)
    baseline = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, _PROTOCOL, baseline)
        exec(compile(source, 'pinned-main-protocol', 'exec'), baseline.__dict__)
        patch.setattr(baseline, 'init_parallel', cpu_v4.protocol.init_parallel)
        expected = _build(baseline, mtp=True)
    actual = _build(cpu_v4.protocol, mtp=True)
    assert type(actual) is type(expected)
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, tensor in actual.state_dict().items():
        _equal(tensor, expected.state_dict()[name])
    ids = torch.tensor([[1, 2, 3, 4]])
    left, right = actual(input_ids=ids, labels=ids), expected(input_ids=ids, labels=ids)
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], list):
            for a, b in zip(left[key], right[key]):
                _equal(a, b)
        else:
            _equal(left[key], right[key])

    left['loss'].backward()
    right['loss'].backward()
    for (name, p), (ref_name, ref_p) in zip(
        actual.named_parameters(), expected.named_parameters()
    ):
        assert name == ref_name
        assert (p.grad is None) == (ref_p.grad is None), name
        if p.grad is not None:
            _equal(p.grad, ref_p.grad)


def test_default_build_does_not_import_packed_modules(cpu_v4, monkeypatch):
    import builtins

    names = [
        'megatron.lite.model.deepseek_v4.lite.packed',
        'megatron.lite.primitive.modules.paired_stream',
        'megatron.lite.primitive.ops.packed_objective',
    ]
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name not in names, f'default path imported {name}'
        return original(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        for name in names:
            patch.delitem(sys.modules, name, raising=False)
        patch.setattr(builtins, '__import__', guarded)
        # Re-import protocol too: catching only build-time imports is insufficient.
        source = Path(cpu_v4.protocol.__file__).read_text()
        module = types.ModuleType(_PROTOCOL)
        patch.setitem(sys.modules, _PROTOCOL, module)
        exec(compile(source, 'default-import-check', 'exec'), module.__dict__)
        patch.setattr(module, 'init_parallel', cpu_v4.protocol.init_parallel)
        model = _build(module)
        model(input_ids=torch.tensor([[1, 2, 3]]), enable_mtp=False)
        assert all(name not in sys.modules for name in names)


def _independent_replay(reference, routers, kwargs, targets, replay_mask):
    from megatron.lite.primitive.modules.router_replay import (
        RouterReplayAction as Action,
    )
    from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy

    outputs = []
    cu = kwargs['packed_seq_params'].cu_seqlens_q_padded.tolist()
    for begin, end in zip(cu, cu[1:]):
        for router, target in zip(routers, targets):
            router.router_replay_action = Action.REPLAY_FORWARD
            router.target_topk_idx = target[begin:end]
            router.target_replay_mask = replay_mask[begin:end]
        outputs.append(
            reference(
                input_ids=kwargs['input_ids'][:, begin:end],
                position_ids=kwargs['position_ids'][:, begin:end],
                enable_mtp=False,
            )
        )
    logits = torch.cat([out['logits'] for out in outputs], 1)
    losses = vocab_parallel_cross_entropy(logits.clone(), kwargs['labels'])
    mask = kwargs['loss_mask']
    return {
        'hidden_states': torch.cat([out['hidden_states'] for out in outputs]),
        'log_probs': -losses,
        'loss': (losses * mask).sum() / mask.sum(),
    }


def test_real_packer_preserves_masks_and_recorded_routes(cpu_v4):
    from megatron.lite.primitive.modules.router_replay import (
        RouterReplayAction as Action,
    )
    from megatron.lite.runtime.contracts import PackedBatch

    packed = _build(cpu_v4.protocol, packed=True)
    reference = _build(cpu_v4.protocol)
    actual_routers, ref_routers = _routers(packed), _routers(reference)
    ids = torch.arange(1, 9)
    batch = PackedBatch(
        ids,
        ids.clone(),
        torch.tensor([3, 5]),
        loss_mask=torch.ones(8),
        r3_replay_mask=torch.tensor([True, True, False, True, True, True, True, False]),
    )
    kwargs = cpu_v4.protocol._prepare_packed_batch_kwargs(packed, batch)
    cu = kwargs['packed_seq_params'].cu_seqlens_q_padded.tolist()
    assert cu[-1] == ids.numel() and kwargs['loss_mask'].sum() == 6
    for r in actual_routers + ref_routers:
        r.router_replay_action = Action.RECORD
    actual = packed(**kwargs)
    ref_outputs, ref_routes = [], [[], []]
    for begin, end in zip(cu, cu[1:]):
        out = reference(
            input_ids=kwargs['input_ids'][:, begin:end],
            position_ids=kwargs['position_ids'][:, begin:end],
            enable_mtp=False,
        )
        ref_outputs.append(out)
        for r, records in zip(ref_routers, ref_routes):
            records.append(r.recorded_topk_idx.clone())
    _equal(
        actual['hidden_states'], torch.cat([o['hidden_states'] for o in ref_outputs])
    )
    for r, records in zip(actual_routers, ref_routes):
        _equal(r.recorded_topk_idx, torch.cat(records))
    routes = torch.nested.as_nested_tensor(
        [torch.ones(3, 2, 1, dtype=torch.long), torch.ones(5, 2, 1, dtype=torch.long)],
        layout=torch.jagged,
    )
    targets = cpu_v4.protocol.pack_routed_experts(packed, batch, routes)
    replay_mask = cpu_v4.protocol.pack_r3_replay_mask(packed, batch)
    assert replay_mask.sum() == 6
    for r, target in zip(actual_routers, targets):
        assert target.shape[0] == cu[-1]
        r.router_replay_action = Action.REPLAY_FORWARD
        r.target_topk_idx, r.target_replay_mask = target, replay_mask
    replayed = packed(**kwargs)
    expected = _independent_replay(reference, ref_routers, kwargs, targets, replay_mask)
    for key, value in expected.items():
        _equal(replayed[key], value)
    assert not torch.equal(replayed['hidden_states'], actual['hidden_states'])
    replayed['loss'].backward()
    assert packed.embed_tokens.embedding.weight.grad.count_nonzero() > 0


def test_padded_training_requires_mask_and_single_batch_row(cpu_v4):
    from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

    model = _build(cpu_v4.protocol, packed=True)
    params = PackedSeqParams.from_cu_seqlens(torch.tensor([0, 4], dtype=torch.int32), 4)
    with pytest.raises(ValueError, match='loss_mask'):
        model(
            input_ids=torch.tensor([[1, 2, 3, 0]]),
            labels=torch.tensor([[2, 3, 0, 0]]),
            packed_seq_params=params,
        )
    with pytest.raises(ValueError, match='batch row'):
        model(input_ids=torch.tensor([[1, 2], [3, 4]]), packed_seq_params=params)


def test_document_recompute_requires_packed_metadata(cpu_v4):
    model = _build(cpu_v4.protocol, packed=True, recompute=True)
    with pytest.raises(ValueError, match='packed_seq_params'):
        model(input_ids=torch.tensor([[1, 2]]), enable_mtp=False)


def test_padded_offsets_are_distinct_from_logical_offsets(cpu_v4):
    from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

    model = _build(cpu_v4.protocol, packed=True)
    reference = _build(cpu_v4.protocol)
    ids = torch.tensor([[1, 2, 0, 3, 4, 5, 6, 0]])
    params = PackedSeqParams(
        cu_seqlens_q=torch.tensor([0, 2, 6], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 3, 8], dtype=torch.int32),
    )
    actual = model(input_ids=ids, packed_seq_params=params)
    expected = torch.cat(
        [
            reference(input_ids=ids[:, :3], enable_mtp=False)['logits'],
            reference(input_ids=ids[:, 3:], enable_mtp=False)['logits'],
        ],
        1,
    )
    _equal(actual['logits'], expected)


def test_runtime_replay_driver_owns_snapshot_queue_lifecycle(cpu_v4):
    from megatron.lite.primitive.modules.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )
    from megatron.lite.runtime.backends.mlite.router_replay import RouterReplayDriver
    from megatron.lite.runtime.contracts import PackedBatch

    model = _build(cpu_v4.protocol, packed=True, recompute=True)
    reference = _build(cpu_v4.protocol)
    ref_routers = _routers(reference)
    handle = SimpleNamespace(_model=model, _extras={'protocol': cpu_v4.protocol})
    driver = RouterReplayDriver(handle, 'replay')
    driver.begin()
    routers = RouterReplay.global_router_replay_instances[:]

    def forward(chunk, batch):
        return chunk(**cpu_v4.protocol._prepare_packed_batch_kwargs(chunk, batch))

    step = driver.wrap(forward)
    try:
        outputs = []
        for route in [1, 0]:
            ids = torch.arange(1, 9)
            batch = PackedBatch(
                ids,
                ids.clone(),
                torch.tensor([3, 5]),
                loss_mask=torch.ones(8),
                r3_replay_mask=torch.tensor(
                    [True, True, False, True, True, True, True, False]
                ),
                routed_experts=torch.nested.as_nested_tensor(
                    [torch.full((3, 2, 1), route), torch.full((5, 2, 1), route)],
                    layout=torch.jagged,
                ),
            )
            outputs.append(step(model, batch))
            kwargs = cpu_v4.protocol._prepare_packed_batch_kwargs(model, batch)
            targets = cpu_v4.protocol.pack_routed_experts(
                model, batch, batch.routed_experts
            )
            replay_mask = cpu_v4.protocol.pack_r3_replay_mask(model, batch)
            expected = _independent_replay(
                reference, ref_routers, kwargs, targets, replay_mask
            )
            for key, value in expected.items():
                _equal(outputs[-1][key], value)
        assert all(
            r.router_replay_action == RouterReplayAction.REPLAY_BACKWARD
            for r in routers
        )
        assert all(len(r.replay_backward_list) == 2 for r in routers)
        for output in reversed(outputs):
            output['loss'].backward()
        assert torch.isfinite(model.embed_tokens.embedding.weight.grad).all()
    finally:
        driver.end()
    assert all(
        not r.replay_backward_list and r.target_topk_idx is None for r in routers
    )
    assert RouterReplay.global_router_replay_instances == []
    assert all(layer.mlp.gate.router_replay is None for layer in model.layers.values())


@pytest.mark.parametrize('option', ['cp', 'pp', 'ep', 'vpp', 'offload', 'recompute'])
def test_opt_in_rejects_unsupported_combinations_before_build(cpu_v4, option):
    from megatron.lite.runtime.contracts import ParallelConfig

    kwargs = {'packed_documents': True, 'mtp_enable': False, 'optimizer': None}
    if option in ('cp', 'pp', 'ep', 'vpp'):
        kwargs['parallel'] = ParallelConfig(**{option: 2})
    else:
        kwargs[option] = ['router']
    with pytest.raises(ValueError, match='V4 document execution'):
        cpu_v4.protocol.build_model(
            _config(), impl_cfg=cpu_v4.protocol.ImplConfig(**kwargs)
        )


def test_opt_in_selects_working_sparse_backend_when_omitted(cpu_v4):
    from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

    model = cpu_v4.protocol.build_model(
        _config(),
        impl_cfg=cpu_v4.protocol.ImplConfig(
            optimizer=None, mtp_enable=False, packed_documents=True
        ),
    ).chunks[0]
    assert all(
        layer.self_attn.self_attn.attention_backend == 'flash'
        for layer in model.layers.values()
    )
    params = PackedSeqParams.from_cu_seqlens(
        torch.tensor([0, 2, 5], dtype=torch.int32), 3
    )
    output = model(input_ids=torch.tensor([[1, 2, 3, 4, 5]]), packed_seq_params=params)
    assert output['logits'].shape[:2] == (1, 5)


@pytest.mark.parametrize('mtp_layers', [0, 1])
@pytest.mark.parametrize(
    'mtp_options',
    [
        {},
        {'mtp_enable': True},
        {'mtp_enable': False, 'mtp_enable_train': True},
        {'mtp_enable': False, 'mtp_detach_encoder': True},
        {'mtp_enable': False, 'mtp_num_layers': 3},
        {'mtp_enable': False, 'num_nextn_predict_layers': 2},
        {'mtp_enable': False, 'mtp_loss_scaling_factor': 0.7},
    ],
)
def test_packed_build_rejects_mtp_configuration(
    cpu_v4, monkeypatch, mtp_layers, mtp_options
):
    config = _config(mtp=bool(mtp_layers))
    original = vars(config).copy()

    def unexpected_init(_):
        pytest.fail('MTP must be rejected before parallel initialization')

    monkeypatch.setattr(cpu_v4.protocol, 'init_parallel', unexpected_init)
    with pytest.raises(ValueError, match='packed.*MTP'):
        cpu_v4.protocol.build_model(
            config,
            impl_cfg=cpu_v4.protocol.ImplConfig(
                optimizer=None, packed_documents=True, **mtp_options
            ),
        )
    assert vars(config) == original


def test_packed_build_rejects_model_mtp_layers(cpu_v4):
    with pytest.raises(ValueError, match='packed.*MTP'):
        cpu_v4.protocol.build_model(
            _config(mtp=True),
            impl_cfg=cpu_v4.protocol.ImplConfig(
                optimizer=None, packed_documents=True, mtp_enable=False
            ),
        )


@pytest.mark.parametrize('temperature', [0.5, 2.0])
@pytest.mark.parametrize('recompute', [False, True])
def test_packed_logits_only_matches_parent_bytes(cpu_v4, temperature, recompute):
    from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

    parent = _build(cpu_v4.protocol)
    packed = _build(cpu_v4.protocol, packed=True, recompute=recompute)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    params = PackedSeqParams.from_cu_seqlens(
        torch.tensor([0, 2, 5], dtype=torch.int32), 3
    )
    expected = torch.cat(
        [
            parent(input_ids=part, temperature=temperature)['logits']
            for part in ids.split([2, 3], dim=1)
        ],
        dim=1,
    )
    actual = packed(input_ids=ids, packed_seq_params=params, temperature=temperature)
    _equal(actual['logits'], expected)
    assert 'loss' not in actual


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize(
    'override', [None, 'flash', 'fused', 'unfused', 'local', 'auto']
)
def test_backend_environment_matches_module_selection(
    cpu_v4, monkeypatch, packed, override
):
    names = ('NVTE_FLASH_ATTN', 'NVTE_FUSED_ATTN', 'NVTE_UNFUSED_ATTN')
    for name in names:
        monkeypatch.setenv(name, 'sentinel')
    expected = override or ('flash' if packed else 'torch')
    env = {
        'torch': ('0', '0', '1'),
        'local': ('0', '0', '1'),
        'unfused': ('0', '0', '1'),
        'flash': ('1', '0', '0'),
        'fused': ('0', '1', '0'),
        'auto': ('1', '1', '1'),
    }[expected]
    observed = []
    original = cpu_v4.model.DeepseekV4Layer.__init__

    def capture(self, *args, **kwargs):
        observed.append(tuple(os.environ[name] for name in names))
        original(self, *args, **kwargs)

    monkeypatch.setattr(cpu_v4.model.DeepseekV4Layer, '__init__', capture)
    model = cpu_v4.protocol.build_model(
        _config(),
        impl_cfg=cpu_v4.protocol.ImplConfig(
            optimizer=None,
            mtp_enable=False,
            packed_documents=packed,
            attention_backend_override=override,
        ),
    ).chunks[0]
    assert observed and all(value == env for value in observed)
    assert tuple(os.environ[name] for name in names) == env
    backends = [
        m.attention_backend for m in model.modules() if hasattr(m, 'attention_backend')
    ]
    assert backends and all(value == expected for value in backends)
