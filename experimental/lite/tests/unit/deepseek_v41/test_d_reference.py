"""Independent pinned-source CPU arithmetic/VJP diagnostics; no GPU claim."""

import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

TOOLS = Path(__file__).resolve().parents[3] / 'tools/deepseek_v41'
sys.path.insert(0, str(TOOLS))
from d_semantics_reference import (
    load_floating_reference,
    original_method,
    validate_cpu_source,
)
from megatron.lite.model.deepseek_v41.lite import attention, candidates, engram
from megatron.lite.model.deepseek_v41.lite.moe import SwiGLUExpert
from test_attention import config
from test_moe import make_router

REFERENCE = Path(
    os.environ.get(
        'DS41_REFERENCE_DIR',
        str(Path(__file__).resolve().parents[2] / 'fixtures/deepseek_v41/reference'),
    )
)


def compare_value_vjp_update(
    actual, expected, params, reference_params, *, atol=2e-5, rtol=2e-5
):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    cotangent = torch.linspace(-0.3, 0.7, actual.numel()).reshape_as(actual)
    ga = torch.autograd.grad((actual * cotangent).sum(), params, allow_unused=True)
    gb = torch.autograd.grad(
        (expected * cotangent).sum(), reference_params, allow_unused=True
    )
    for p, q, a, b in zip(params, reference_params, ga, gb):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
            torch.testing.assert_close(p - 0.01 * a, q - 0.01 * b, atol=atol, rtol=rtol)


def test_all_40_attention_layers_against_pinned_floating_methods():
    from dataclasses import replace

    torch.manual_seed(4101)
    c = replace(config(), candidate_blocks=3)
    ns = load_floating_reference(REFERENCE)
    args = ns['ModelArgs'](
        n_layers=40,
        dim=c.dim,
        n_heads=c.heads,
        head_dim=c.head_dim,
        rope_head_dim=c.rope_dim,
        q_lora_rank=c.q_rank,
        o_lora_rank=c.o_rank,
        o_groups=c.groups,
        index_n_heads=c.index_heads,
        index_head_dim=c.index_dim,
        index_topk=c.topk,
        window_size=c.window,
        norm_eps=c.eps,
        max_batch_size=1,
        max_seq_len=12,
        original_seq_len=c.original_length,
        rope_factor=c.factor,
        rope_theta=c.rope_theta,
        compress_rope_theta=c.compress_rope_theta,
        beta_fast=c.beta_fast,
        beta_slow=c.beta_slow,
        compress_ratios=(0, 0) + (2,) * 18 + (1,) * 20,
        kv_source_layers=(2, 8, 14, 20),
        index_source_layers=(2, 8, 14, 20, 24, 28, 32, 36),
        candidate_source_layer=20,
        candidate_topk_blocks=c.candidate_blocks,
        candidate_block_size=c.block_size,
    )
    state = attention.AttentionState()
    native_params, ref_params, inputs, ref_inputs, outputs, refs = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for layer in range(40):
        native = attention.CSA2Attention(c, layer).float()
        official = ns['Attention'](layer, args).float()
        if official.indexer is not None:
            official.indexer.requires_grad_(False)
        official.load_state_dict(native.state_dict(), strict=True)
        x = torch.randn(1, 9, c.dim, requires_grad=True)
        rx = x.detach().clone().requires_grad_()
        actual, state = native(x, state)
        expected = official(rx, 0)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        if state.main_kv is not None:
            torch.testing.assert_close(
                state.main_kv, ns['shared_attn'].compress_kv[:, : 9 // native.ratio]
            )
            torch.testing.assert_close(
                state.indices,
                torch.where(
                    ns['shared_attn'].topk_idxs >= 0,
                    ns['shared_attn'].topk_idxs - 9,
                    -1,
                ),
                atol=0,
                rtol=0,
            )
        np = dict(native.named_parameters())
        rp = dict(official.named_parameters())
        for key in np:
            if np[key].requires_grad:
                native_params.append(np[key])
                ref_params.append(rp[key])
        inputs.append(x)
        ref_inputs.append(rx)
        outputs.append(actual)
        refs.append(expected)
    compare_value_vjp_update(
        torch.stack(outputs),
        torch.stack(refs),
        inputs + native_params,
        ref_inputs + ref_params,
        atol=1e-4,
        rtol=1e-4,
    )


def test_modality_router_against_original_gate_and_parameter_gradients():
    torch.manual_seed(4102)
    router = make_router()
    ns = load_floating_reference(REFERENCE)
    args = ns['ModelArgs'](
        dim=3,
        n_routed_experts=3,
        n_activated_experts=2,
        gate_temp=0.7,
        route_scale=1.5,
        vision_n_layers=1,
    )
    ref = ns['Gate'](0, args)
    with torch.no_grad():
        ref.weight.copy_(router.router.gate.weight)
        ref.bias.copy_(router.bias)
        ref.bias_vl.copy_(router.bias_vl)
    x = torch.tensor([[2.0, 1.0, -1.0], [-1.0, 1.0, 2.0]], requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    image = torch.tensor([False, True])
    a, ia, _ = router(x, image)
    b, ib = ref(rx, image)
    order = ib.argsort(-1)
    torch.testing.assert_close(ia, ib.gather(-1, order))
    compare_value_vjp_update(
        a, b.gather(-1, order), [x, router.router.gate.weight], [rx, ref.weight]
    )


def test_engram_original_method_value_all_parameter_vjps_and_updates():
    torch.manual_seed(4103)
    module = engram.Engram(4, 2, nn.Embedding(24, 3), nn.Linear(6, 12, bias=False))
    ref = copy.deepcopy(module)
    ref.hc_mult = 2
    ref.clamp_value = 1e-6
    ref.forward = original_method(REFERENCE, 'Engram', 'forward').__get__(ref)
    x = torch.randn(1, 5, 2, 4, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    ids = torch.tensor([[[1, 12], [2, 13], [3, 14], [4, 15], [5, 16]]])
    mask = torch.tensor([[True, False, True, True, True]])
    compare_value_vjp_update(
        module(x, ids, mask),
        ref(rx, ids, mask),
        [x, *module.parameters()],
        [rx, *ref.parameters()],
    )


def test_hash_normalization_layout_and_resets_against_original_module():
    validate_cpu_source(REFERENCE, 'engram.py')
    spec = importlib.util.spec_from_file_location(
        'ds41_pinned_hash', REFERENCE / 'engram.py'
    )
    ref = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ref
    spec.loader.exec_module(ref)

    class Tokenizer:
        words = ['<pad>', ' The', 'the', 'THE', 'É', 'e', ' ', '', '\ufffd', 'ＦＯＯ']

        def __init__(self):
            self.backend_tokenizer = self

        def __len__(self):
            return len(self.words)

        def decode(self, ids, **kwargs):
            return self.words[ids[0]]

        def id_to_token(self, token):
            return '<byte>'

    tokenizer = Tokenizer()
    mapping, vocab = engram.build_compressed_token_map(tokenizer)
    assert (mapping, vocab) == ref.build_compressed_token_map(tokenizer)
    assert mapping[1] == mapping[2] == mapping[3] and mapping[4] == mapping[5]
    args = SimpleNamespace(
        engram_layer_ids=(1, 14),
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_vocab_size=31,
        engram_num_embeddings=(999, 999),
        engram_head_dim=4,
        engram_compressed_vocab_size=vocab,
        engram_pad_id=0,
        max_batch_size=1,
        max_seq_len=8,
    )
    layout = ref.EngramLayout.from_args(args)
    primes = engram.prime_buckets((1, 14), 4, 2, 31)
    multipliers = engram.hash_multipliers((1, 14), 4, vocab)
    torch.testing.assert_close(primes, torch.tensor(layout.primes), atol=0, rtol=0)
    torch.testing.assert_close(
        multipliers, ref.compute_hash_multipliers((1, 14), 4, vocab), atol=0, rtol=0
    )
    native = engram.NgramHash(mapping, 0, multipliers, primes)
    official = ref.NgramHashState(args, layout, tokenizer)
    ids = torch.tensor([[1, 2, 4, 6, 5, 9, 3, 8]])
    mask = torch.tensor([[True, True, False, True, True, True, False, True]])
    torch.testing.assert_close(
        native(ids, mask), official(ids, 0, mask), atol=0, rtol=0
    )
    expected = torch.cat(
        [official(ids[:, :3], 0, mask[:, :3]), official(ids[:, 3:], 0, mask[:, 3:])], 1
    )
    torch.testing.assert_close(
        native(ids, mask, cu_seqlens=torch.tensor([0, 3, 8])), expected, atol=0, rtol=0
    )


def test_candidates_against_original_function_with_partial_and_empty_prefixes():
    fn = load_floating_reference(REFERENCE)['select_candidate_blocks']
    scores = torch.tensor([[[1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0]]]).expand(1, 8, 7)
    lengths = torch.arange(8)[:, None]
    visible = torch.arange(7) < lengths
    for budget in (1, 2, 4):
        expected = fn(scores.masked_fill(~visible, -torch.inf), lengths, budget, 2)
        actual = candidates.candidate_blocks(
            scores, lengths, topk_blocks=budget, block_size=2
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_expert_clamping_and_weight_placement_original_method_vjp():
    torch.manual_seed(4104)
    native = SwiGLUExpert(
        nn.Linear(4, 8, bias=False),
        nn.Linear(8, 4, bias=False),
        nn.Linear(4, 8, bias=False),
        swiglu_limit=0.4,
    )
    ref = copy.deepcopy(native)
    ref.forward = original_method(REFERENCE, 'Expert', 'forward').__get__(ref)
    x = torch.randn(3, 4, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    weights = torch.tensor([[0.2], [0.7], [1.1]], requires_grad=True)
    rw = weights.detach().clone().requires_grad_()
    compare_value_vjp_update(
        native(x, weights),
        ref(rx, rw),
        [x, weights, *native.parameters()],
        [rx, rw, *ref.parameters()],
    )


def test_mhc_scalar_kernel_equations_all_coefficients_vjp_and_update():
    from megatron.lite.model.deepseek_v41.lite.block import HCMixes

    torch.manual_seed(4105)
    native = HCMixes(3, 2, hc_eps=0.01, iterations=3)
    ref = copy.deepcopy(native)
    x = torch.randn(1, 2, 2, 3, requires_grad=True)
    rx = x.detach().clone().requires_grad_()

    # Scalar equations independently transcribe pinned kernel.py:407-462.
    # This tests the CPU port derivative, not the compiled kernel's backward.
    def scalar(h):
        result = []
        for token in h.flatten(0, 1):
            flat = token.flatten().float()
            values = [
                (flat * row).sum() / (flat.square().mean() + ref.norm_eps).sqrt()
                for row in ref.fn
            ]
            pre = torch.stack(
                [
                    torch.sigmoid(values[i] * ref.scale[0] + ref.base[i]) + ref.hc_eps
                    for i in range(2)
                ]
            )
            post = torch.stack(
                [
                    2 * torch.sigmoid(values[2 + i] * ref.scale[1] + ref.base[2 + i])
                    for i in range(2)
                ]
            )
            rows = [
                torch.stack(
                    [
                        values[4 + 2 * i + j] * ref.scale[2] + ref.base[4 + 2 * i + j]
                        for j in range(2)
                    ]
                ).softmax(0)
                + ref.hc_eps
                for i in range(2)
            ]
            matrix = [[rows[i][j] for j in range(2)] for i in range(2)]

            def columns(m):
                return [
                    [
                        m[i][j] / (sum(m[k][j] for k in range(2)) + ref.hc_eps)
                        for j in range(2)
                    ]
                    for i in range(2)
                ]

            matrix = columns(matrix)
            for _ in range(ref.iterations - 1):
                matrix = [[v / (sum(row) + ref.hc_eps) for v in row] for row in matrix]
                matrix = columns(matrix)
            result.append(
                torch.cat(
                    [
                        pre,
                        post,
                        torch.stack([torch.stack(row) for row in matrix]).flatten(),
                    ]
                )
            )
        return torch.stack(result).reshape(1, 2, 8)

    pre, post, comb = native(x)
    compare_value_vjp_update(
        torch.cat([pre, post, comb.flatten(-2)], -1),
        scalar(rx),
        [x, *native.parameters()],
        [rx, *ref.parameters()],
    )
