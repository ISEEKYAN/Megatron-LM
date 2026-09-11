# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Block-FP8 quantize/dequantize round-trip fidelity.

Validates numerical fidelity only (the quantize->dequantize round-trip error stays
within the E4M3 blockwise magnitude); it does not validate layout, axis order, or
fusion. Zero GPU, pure tensors. DS4 routed experts use expert_dtype=fp8 ->
block_fp8.quantize_block_fp8 with scale_format="float32", so every case here exercises the
float32 scale branch.
"""

from __future__ import annotations

import pytest
import torch
from megatron.lite.primitive.quantization import block_fp8

# E4M3 has 3 mantissa bits -> per-value resolution ~2^-3 = 12.5%. With blockwise
# per-128x128 absmax scaling, the overall (Frobenius) energy relative error should
# be ~2^-3/sqrt(3), i.e. a few percent. We gate on 6%; element-wise max-rel is not
# a criterion (near-zero elements inside a block have naturally large rel error).
FROBENIUS_TOL = 0.06


def _frobenius_rel_err(restored: torch.Tensor, source: torch.Tensor) -> float:
    return (
        torch.linalg.vector_norm(restored.float() - source.float())
        / torch.linalg.vector_norm(source.float())
    ).item()


def _roundtrip(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight, scale = block_fp8.quantize_block_fp8(
        source, (128, 128), scale_format="float32"
    )
    restored = block_fp8.dequantize_block_fp8(weight, scale, (128, 128))
    return weight, scale, restored


def test_roundtrip_extreme_negative_and_near_zero_values() -> None:
    """Tensor with E4M3 extremes / negatives / near-zero / exact-zero round-trips within fp8 magnitude."""
    torch.manual_seed(0)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    source = torch.empty(256, 256)
    # Mix in block-level large magnitudes (+/- near fp8_max), tiny values, exact 0,
    # and a regular small-weight scale.
    source[:128, :128] = torch.linspace(-fp8_max, fp8_max, 128 * 128).reshape(128, 128)
    source[:128, 128:] = torch.full((128, 128), 1e-4)
    source[128:, :128] = torch.randn(128, 128) * 0.02
    block = torch.randn(128, 128) * 0.02
    block[0, 0] = fp8_max  # one large outlier in the block sets the scale
    block[1, 1] = -fp8_max
    block[2, 2] = 0.0  # exact zero must round-trip back to zero
    source[128:, 128:] = block

    weight, scale, restored = _roundtrip(source)

    assert weight.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    assert scale.shape == (2, 2)  # 256/128 x 256/128
    assert torch.isfinite(restored).all(), "dequant produced NaN/Inf"
    # Exact-zero elements must restore to 0 (finite scale, quantization preserves 0).
    assert restored[128 + 2, 128 + 2].abs().item() == 0.0
    rel = _frobenius_rel_err(restored, source)
    assert rel < FROBENIUS_TOL, f"Frobenius rel err {rel:.4f} >= {FROBENIUS_TOL}"


@pytest.mark.parametrize(
    "shape,label",
    [
        ((512, 1024), "w1_gate_up_I_by_H"),  # expert gate/up: [I, H]
        ((1024, 512), "w2_down_H_by_I"),  # expert down:    [H, I]
    ],
)
def test_roundtrip_ds4_expert_shapes(shape: tuple[int, int], label: str) -> None:
    """DS4 routed-expert weight shapes (w1[I,H]/w2[H,I]) round-trip under a realistic weight distribution."""
    torch.manual_seed(1234)
    # Realistic weight-like distribution: small-scale normal + a few outliers (which set each block's scale).
    source = torch.randn(*shape) * 0.02
    outlier_mask = torch.rand(*shape) < 0.001
    source = torch.where(outlier_mask, torch.sign(source) * 5.0, source)

    weight, scale, restored = _roundtrip(source)

    assert (
        weight.shape == source.shape
    ), f"{label}: quantization changed the shape (suspect axis/layout error)"
    assert scale.shape == (shape[0] // 128, shape[1] // 128)
    assert torch.isfinite(restored).all()
    rel = _frobenius_rel_err(restored, source)
    assert (
        rel < FROBENIUS_TOL
    ), f"{label}: Frobenius rel err {rel:.4f} >= {FROBENIUS_TOL}"


def test_roundtrip_is_deterministic() -> None:
    """The same input quantized/dequantized twice is bit-identical (no randomness; usable as a CI baseline)."""
    source = torch.randn(128, 128) * 0.05
    _, _, a = _roundtrip(source)
    _, _, b = _roundtrip(source)
    assert torch.equal(a, b)


@pytest.fixture
def _engram_cpu():
    with torch.device('cpu'):
        yield


def _engram_state(trainable=True, width=32):
    from megatron.lite.model.deepseek_v41.lite import table_state
    from megatron.lite.primitive.modules import engram_lookup

    table = engram_lookup.ShardedEngramTable(
        torch.full((4, width), 256.0).to(torch.float8_e4m3fn),
        torch.full((4, width // 32), 1 / 256).to(torch.float8_e8m0fnu),
        engram_lookup.RowLookup((0, 4)),
        trainable=trainable,
    )
    return table_state.EngramTableState(table)


def _engram_ready(state, step, ids, coefficients):
    from megatron.lite.model.deepseek_v41.lite import prefetch

    ids = torch.tensor(ids, dtype=torch.int64)
    batch = prefetch.EngramPrefetch(state).start(step, {'mb': ids})
    result = batch.view('mb')(ids)
    if state.table.master is not None:
        (result.float() * torch.tensor(coefficients)[:, None]).sum().backward()
    batch.flush()


@pytest.mark.parametrize(
    'rows,expected', [(7, ((0, 3), (3, 5), (5, 7))), (1, ((0, 1), (1, 1), (1, 1)))]
)
def test_engram_existing_rank_ownership(rows, expected, _engram_cpu):
    from megatron.lite.model.deepseek_v41.lite.parallel import EngramLayout

    layout = EngramLayout(
        rows,
        ((0, 2, 4), (1, 3, 5)),
        world_size=6,
        aliases={'reuse': 'shadow', 'shadow': 'owner'},
    )
    assert layout.row_intervals == expected
    assert layout.replica_groups == ((0, 1), (2, 3), (4, 5))
    assert sorted(
        i for r in range(6) for i in range(*layout.optimizer_interval(r))
    ) == list(range(rows))
    assert layout.canonical_name('reuse') == 'owner'
    for ranks, aliases in [
        (((0, 0),), {}),
        (((0, 6),), {}),
        (((0,),), {'a': 'b', 'b': 'a'}),
    ]:
        with pytest.raises(ValueError):
            EngramLayout(rows, ranks, world_size=6, aliases=aliases)


@pytest.mark.parametrize('trainable', [False, True])
def test_engram_prefetch_tags_bytes_and_delayed_fp32_return(
    trainable, monkeypatch, _engram_cpu
):
    from megatron.lite.model.deepseek_v41.lite import prefetch

    state = _engram_state(trainable, width=256)
    calls, lookup = [], state.table.lookup_fp8

    def counted(ids):
        calls.append(ids.clone())
        return lookup(ids)

    monkeypatch.setattr(state.table, 'lookup_fp8', counted)
    ids = {'a': torch.tensor([[2, 0, 2] * 8]), 'b': torch.tensor([[1, 2, 1] * 8])}
    batch = prefetch.EngramPrefetch(state).start(7, ids)
    with pytest.raises(RuntimeError, match='missing'):
        batch.flush()
    with pytest.raises(RuntimeError, match='pending'):
        state.step(torch.clone, lr=0.1)
    with pytest.raises(RuntimeError, match='active'):
        state.state_dict()
    for key in ('b', 'a'):
        view = batch.view(key)
        values, scales, master = view.lookup_fp8(ids[key])
        assert values.flatten(-2).shape == (1, 6144)
        assert torch.equal(
            values.view(torch.uint8), state.table.weight.view(torch.uint8)[ids[key]]
        )
        assert torch.equal(
            scales.view(torch.uint8), state.table.scale.view(torch.uint8)[ids[key]]
        )
        if trainable:
            coefficient = 1 if key == 'a' else 2**-10
            (view(ids[key]).float() * coefficient).sum().backward()
            with pytest.raises(RuntimeError, match='duplicate'):
                batch.return_gradient(
                    prefetch.GradientTag(7, 0, key), torch.ones_like(master)
                )
            with pytest.raises(RuntimeError, match='tag'):
                batch.return_gradient(
                    prefetch.GradientTag(7, 1, key), torch.ones_like(master)
                )
    if trainable:
        assert state.table.master.grad is None and not torch.count_nonzero(
            state.main_grad
        )
    batch.flush()
    assert len(calls) == 1
    if trainable:
        expected = torch.tensor([8.0, 16 * 2**-10, 16 + 8 * 2**-10, 0.0])[
            :, None
        ].expand(4, 256)
        torch.testing.assert_close(state.main_grad, expected, atol=0, rtol=0)
        assert not torch.equal(expected, expected.bfloat16().float())
    else:
        assert state.main_grad is state.momentum is None
    with pytest.raises(RuntimeError, match='closed'):
        batch.view('a')
    state.step(torch.clone, lr=0.001)


@pytest.mark.parametrize('restore', [False, True])
def test_engram_full_row_momentum_and_restored_trajectory(restore, _engram_cpu):
    state = _engram_state()
    _engram_ready(state, 0, [2, 2], [1.0, 3.0])
    state.step(torch.clone, lr=0.1)
    torch.testing.assert_close(state.table.master[2], torch.full((32,), 0.961))
    if restore:
        saved = state.state_dict()
        state = _engram_state()
        state.load_state_dict(saved)
        saved['master'].zero_()
    _engram_ready(state, 1, [0], [2.0])
    state.step(torch.clone, lr=0.1)
    torch.testing.assert_close(state.momentum[2], torch.full((32,), 0.19))
    torch.testing.assert_close(state.table.master[2], torch.full((32,), 0.94295))
    assert state.version == 2
    # Tiny increments survive even when the first FP8 publication is unchanged.
    state = _engram_state()
    before = state.table.weight.view(torch.uint8).clone()
    for step in range(40):
        _engram_ready(state, step, [0], [1.0])
        state.step(torch.clone, lr=0.001, beta=0.0)
        if step == 0:
            assert torch.equal(before, state.table.weight.view(torch.uint8))
    torch.testing.assert_close(
        state.table.master[0], torch.full((32,), 0.96), atol=1e-6, rtol=0
    )
    assert not torch.equal(before[0], state.table.weight.view(torch.uint8)[0])


@pytest.mark.parametrize('mode', ['frozen', 'skip', 'nan', 'publication_failure'])
def test_engram_publication_transaction(mode, monkeypatch, _engram_cpu):
    import megatron.lite.model.deepseek_v41.lite.table_state as module

    state = _engram_state(mode != 'frozen')
    before = state.state_dict()
    _engram_ready(state, 0, [0], [float('nan') if mode == 'nan' else 2.0])

    def fail(*args, **kwargs):
        raise RuntimeError('publication_failure')

    with monkeypatch.context() as m:
        if mode in ('publication_failure', 'frozen'):
            m.setattr(module, 'quantize_block_fp8', fail)
        if mode == 'publication_failure':
            with pytest.raises(RuntimeError, match=mode):
                state.step(torch.clone, lr=0.1)
        else:
            assert not state.step(torch.clone, lr=0.1, skip=mode == 'skip')
    assert state.version == 0
    for key in ('weight', 'scale', 'master'):
        value = getattr(state.table, key)
        if value is not None:
            assert torch.equal(value.view(torch.uint8), before[key].view(torch.uint8))
    if mode == 'publication_failure':
        assert state.step(torch.clone, lr=0.1)
        assert state.version == 1


@pytest.mark.parametrize('damage', [None, 'gap', 'overlap', 'payload'])
def test_engram_bounded_row_loader(tmp_path, damage, monkeypatch, _engram_cpu):
    import hashlib
    from types import SimpleNamespace

    from megatron.lite.model.deepseek_v41.lite import checkpoint

    name = 'layers.0.engram.embed.weight'
    values, scales = torch.arange(7 * 256).byte().reshape(7, 256), torch.full(
        (7, 8), 127, dtype=torch.uint8
    )
    path, entries, offset = tmp_path / 'rows', {}, 0
    with path.open('wb') as f:
        for key, tensor, dtype in (
            (name, values, 'F8_E4M3'),
            (name[:-6] + 'scale', scales, 'F8_E8M0'),
        ):
            raw = tensor.numpy().tobytes()
            f.write(raw)
            entries[key] = SimpleNamespace(
                release_key=key,
                shape=tuple(tensor.shape),
                dtype=dtype,
                source_shard=str(path),
                offset=offset,
                byte_length=len(raw),
                payload_digest=hashlib.sha256(raw).hexdigest(),
            )
            offset += len(raw)
    intervals = {'gap': ((0, 3), (4, 7)), 'overlap': ((0, 4), (3, 7))}.get(
        damage, ((0, 3), (3, 7))
    )
    if damage == 'payload':
        with path.open('r+b') as f:
            f.seek(values.numel() - 1)
            f.write(b'\x00')
    if damage:
        with pytest.raises(ValueError):
            checkpoint.load_engram_rows(
                SimpleNamespace(entries=entries),
                name,
                intervals=intervals,
                rank=0,
                device='cpu',
                chunk_rows=2,
            )
    else:
        chunks = [
            checkpoint.load_engram_rows(
                SimpleNamespace(entries=entries),
                name,
                intervals=intervals,
                rank=r,
                device='cpu',
                chunk_rows=2,
            )
            for r in range(2)
        ]
        from megatron.lite.primitive.modules import engram_lookup

        table = checkpoint.load_engram_table(
            SimpleNamespace(entries=entries),
            name,
            engram_lookup.RowLookup((0, 7)),
            device='cpu',
            chunk_rows=2,
        )
        assert torch.equal(table.weight.view(torch.uint8), values)
        assert table.master is None
        for i, expected in enumerate((values, scales)):
            assert torch.equal(
                torch.cat([chunk[i].view(torch.uint8) for chunk in chunks]), expected
            )


@pytest.mark.parametrize(
    'tokens,trainable', [(0, True), (3, True), (16, False), (16, True)]
)
def test_engram_published_fp8_projection(tokens, trainable, monkeypatch, _engram_cpu):
    import os

    import megatron.lite.primitive.quantization.engram_fp8 as fp8

    values = (
        (torch.arange(tokens * 64).reshape(tokens, 64) % 3 - 1)
        .float()
        .to(torch.float8_e4m3fn)
    )
    scales = (2.0 ** (torch.arange(tokens * 2).reshape(tokens, 2) % 3 - 1)).to(
        torch.float8_e8m0fnu
    )
    # Distinct output-block scales make a one-column scale roll observable.
    weight = (
        torch.tensor([[1.0, 2.0], [4.0, 8.0]])
        .repeat_interleave(32, 0)
        .repeat_interleave(32, 1)
        .requires_grad_()
    )
    # No local/login-node CUDA execution; GPU arithmetic is exercised by Slurm.
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match='CUDA'):
            fp8.published_fp8_linear(values, scales, weight)
        return
    values, scales = values.cuda(), scales.cuda()
    weight = weight.detach().cuda().requires_grad_()
    master = (
        torch.ones(tokens, 64, device='cuda', requires_grad=True) if trainable else None
    )
    native = fp8._fp8_gemm
    calls = []

    def observe(a, a_scale, b, b_scale):
        assert torch.equal(a.view(torch.uint8), values.view(torch.uint8))
        assert torch.equal(a_scale.view(torch.uint8), scales.view(torch.uint8))
        calls.append(a.shape)
        return native(a, a_scale, b, b_scale)

    monkeypatch.setattr(fp8, '_fp8_gemm', observe)
    result = fp8.published_fp8_linear(
        values, scales, weight, master=master, output_dtype=torch.float32
    )
    assert calls == [values.shape]
    decoded = values.float() * scales.float().repeat_interleave(32, -1)
    torch.testing.assert_close(result, decoded @ weight.T, atol=0, rtol=0)
    result.sum().backward()
    torch.testing.assert_close(
        weight.grad, torch.ones_like(result).T @ decoded, atol=0, rtol=0
    )
    if trainable:
        torch.testing.assert_close(
            master.grad, weight.sum(0).expand_as(master), atol=0, rtol=0
        )


def test_sinkhorn_algorithm1_restarts_from_current_n_and_preserves_momentum():
    """Algorithm 1 has 11 fresh-N passes, no decay, and a 5x Engram LR route."""
    from megatron.lite.primitive.optimizers.sinkhorn import algorithm1_update

    weight = torch.ones(2, 2)
    momentum = torch.zeros_like(weight)
    gradient = torch.ones_like(weight)
    first, momentum, nesterov = algorithm1_update(
        weight, momentum, gradient, lr=1.0, multiplier=5.0
    )
    # Uniform N normalizes to ones and sqrt(n) restores a unit direction.
    torch.testing.assert_close(first, torch.full_like(weight, 0.1), atol=1e-6, rtol=0)
    torch.testing.assert_close(momentum, torch.full_like(weight, 0.05))
    torch.testing.assert_close(nesterov, torch.full_like(weight, 0.0975))
    second, momentum, _ = algorithm1_update(
        first, momentum, torch.zeros_like(weight), lr=1.0, multiplier=5.0
    )
    assert torch.all(second < first)  # historical momentum remains active.
    torch.testing.assert_close(momentum, torch.full_like(weight, 0.0475))
