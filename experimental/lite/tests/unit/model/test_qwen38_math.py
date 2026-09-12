import pytest
import torch
from megatron.lite.model.qwen3_8_flash_next.engram import (
    Qwen3_8_FlashNextNGramEmbedding as NgramEmbedding,
)
from megatron.lite.model.qwen3_8_flash_next.math import (
    Qwen3_8_FlashNextHyperConnection as HyperConnection,
)
from megatron.lite.model.qwen3_8_flash_next.math import qsa_routes, sparse_attention


def test_hash_constants_and_document_reset():
    m = NgramEmbedding(None)
    ids = torch.tensor([[7, 9, 248044, 11, 13]])
    got = m.hash_ids(ids)
    primes = [
        20000003,
        20000023,
        20000033,
        20000047,
        20000059,
        20000063,
        20000069,
        20000077,
        20000081,
        20000093,
        20000107,
        20000147,
        20000153,
        20000159,
        20000161,
        20000171,
    ]
    expected = []
    for t in range(5):
        previous = max([-1] + [i for i in range(t) if ids[0, i] == 248044])
        tokens = [int(ids[0, t - s]) if t - s > previous else 248044 for s in range(3)]
        row, offset = [], 0
        for h, p in enumerate(primes):
            x = tokens[0] * 23703573157769 ^ tokens[1] * 20109073645365
            if h >= 8:
                x ^= tokens[2] * 8052911324071
            row.append(x % p + offset)
            offset += p
        expected.append(row)
    assert got.tolist() == [expected], 'HASH_CONSTANT_PRIME_EOS'
    packed = m.hash_ids(torch.tensor([[7, 9, 11, 13]]), torch.tensor([0, 2, 4]))
    separate = torch.cat(
        [m.hash_ids(torch.tensor([[7, 9]])), m.hash_ids(torch.tensor([[11, 13]]))], 1
    )
    assert torch.equal(packed, separate), 'HASH_PACKED_RESET'


def test_lookup_boundary():
    with expect_guard('ROW_LOOKUP_GATHER_ROWS_REQUIRED'):
        NgramEmbedding(None)(torch.tensor([[1]]))


def test_hc_read_write_and_gradient():
    m = HyperConnection(2, 4, 3).double()
    x = torch.arange(1.0, 17.0, dtype=torch.double).reshape(1, 2, 8).requires_grad_()
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    y, residual = m.mix(x)
    streams = x.reshape(1, 2, 4, 2)
    normalized = streams * torch.rsqrt(streams.square().mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(y, normalized.mean(-2) * 0.5), 'HC_READ_MEAN_GATE'
    out = m.combine(y, residual)
    assert torch.allclose(
        out, x + y.unsqueeze(-2).expand(-1, -1, 4, -1).flatten(-2)
    ), 'HC_WRITE_GATE'
    out.sum().backward()
    assert torch.isfinite(x.grad).all(), 'HC_GRADIENT'


def test_qsa_sparse_tail_padding_and_offset():
    q = torch.ones(1, 7, 1, 1)
    k = torch.tensor([[[[1.0]], [[3.0]], [[2.0]]]])
    routes = qsa_routes(q, k, torch.tensor([7]), token_budget=2, compress_ratio=2)
    assert routes[0, 5].tolist() == [2, 3, -1], 'QSA_TOP_BLOCK'
    assert routes[0, 6].tolist() == [2, 3, 6], 'QSA_CAUSAL_TAIL'
    local = qsa_routes(
        q[:, 4:], k, torch.tensor([7]), token_budget=2, compress_ratio=2, offset=4
    )
    assert torch.equal(local, routes[:, 4:]), 'QSA_CONTIGUOUS_OFFSET'
    padded = qsa_routes(q, k, torch.tensor([5]), token_budget=2, compress_ratio=2)
    assert (padded[:, 5:] == -1).all(), 'QSA_PADDING'


def test_sparse_attention_value_and_backward():
    q = torch.ones(1, 2, 2, 1, requires_grad=True)
    k = torch.zeros(1, 3, 1, 1, requires_grad=True)
    v = torch.tensor([[[[2.0]], [[4.0]], [[99.0]]]], requires_grad=True)
    y = sparse_attention(q, k, v, torch.tensor([[[0, 1], [-1, -1]]]))
    assert y.flatten().tolist() == [3.0, 3.0, 0.0, 0.0], 'QSA_SELECTED_VALUES_EMPTY'
    y.sum().backward()
    assert v.grad.flatten().tolist() == [1.0, 1.0, 0.0], 'QSA_SELECTED_GRADIENT'


@pytest.mark.parametrize(
    'case,tag',
    [
        ('ids', 'HASH_IDS_INTEGER_BS'),
        ('packed', 'HASH_PACKED_BOUNDARIES'),
        ('hc_dim', 'HC_DIMENSIONS'),
        ('hc_width', 'HC_STREAM_WIDTH'),
        ('hc_read', 'HC_READ_ONLY'),
        ('qk', 'QSA_QK_SHAPE'),
        ('budget', 'QSA_BUDGET'),
        ('length', 'QSA_LENGTHS'),
        ('blocks', 'QSA_MISSING_BLOCKS'),
        ('attn', 'QSA_ATTENTION_SHAPE'),
        ('route', 'QSA_ROUTE_RANGE'),
    ],
)
def test_individual_guards(case, tag):
    q = torch.ones(1, 4, 2, 2)
    k = torch.ones(1, 2, 1, 2)
    with expect_guard(tag):
        if case == 'ids':
            NgramEmbedding(None).hash_ids(torch.ones(1, 2))
        elif case == 'packed':
            NgramEmbedding(None).hash_ids(
                torch.ones(1, 2, dtype=torch.long), torch.tensor([0, 1])
            )
        elif case == 'hc_dim':
            HyperConnection(2, 1, 3)
        elif case == 'hc_width':
            HyperConnection(2, 4, 3).mix(torch.ones(1, 3))
        elif case == 'hc_read':
            HyperConnection(2, 4, 3, write=False).combine(None, None)
        elif case == 'qk':
            qsa_routes(q, k.repeat(1, 1, 2, 1), torch.tensor([4]))
        elif case == 'budget':
            qsa_routes(q, k, torch.tensor([4]), token_budget=3)
        elif case == 'length':
            qsa_routes(q, k, torch.tensor([-1]))
        elif case == 'blocks':
            qsa_routes(q, k[:, :0], torch.tensor([4]))
        elif case == 'attn':
            sparse_attention(
                q, k, k[:, :, :, :1], torch.zeros(1, 4, 1, dtype=torch.long)
            )
        elif case == 'route':
            sparse_attention(q, k, k, torch.full((1, 4, 1), 3, dtype=torch.long))


def test_hc_nonzero_gates_reference():
    torch.manual_seed(38)
    m = HyperConnection(3, 4, 2).double()
    x = torch.randn(2, 12, dtype=torch.double)
    z = x.reshape(2, 4, 3)
    n = (z / torch.sqrt(z.square().mean(-1, keepdim=True) + 1e-6)).reshape(2, 12) * (
        1 + m.hc_norm.weight
    )
    down = n @ m.input_mix_weight_down.weight.T / 4
    gate = torch.sigmoid((down * torch.sigmoid(down)) @ m.input_mix_weight_up.weight.T)
    expected = (n * gate).reshape(2, 4, 3).sum(1) / 4
    y, residual = m.mix(x)
    assert torch.allclose(y, expected, atol=1e-12), 'HC_NONZERO_READ'
    injection = torch.sigmoid(n @ m.block_inject_weight.weight.T / 4) * 2
    assert torch.allclose(
        m.combine(y, residual),
        x + (injection[:, :, None] * expected[:, None, :]).reshape(2, 12),
        atol=1e-12,
    ), 'HC_NONZERO_WRITE'


def test_qsa_module_packed_padded_gradient():
    from megatron.lite.model.qwen3_8_flash_next import config as qwen_config

    Qwen38Config = qwen_config.Qwen3_8_FlashNextTextConfig
    from megatron.lite.model.qwen3_8_flash_next import qsa as qwen_qsa

    QSA = qwen_qsa.Qwen3_8_FlashNextQSAAttention

    c = Qwen38Config(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        indexer_n_heads=2,
        indexer_head_dim=2,
        indexer_compress_ratio=2,
        indexer_budget=2,
    )
    torch.manual_seed(38)
    m = QSA(c)
    x = torch.randn(1, 7, 4, requires_grad=True)
    angles = torch.zeros(1, 7, 1, 2)
    packed = m(x, angles, cu_seqlens=torch.tensor([0, 3, 7]))
    expected = torch.cat([m(x[:, :3], angles[:, :3]), m(x[:, 3:], angles[:, 3:])], 1)
    assert torch.equal(packed, expected), 'QSA_PACKED_DOCUMENT_RESET'
    padded = m(x, angles, lengths=torch.tensor([3]))
    assert (
        torch.allclose(padded[:, :3], m(x[:, :3], angles[:, :3]), atol=1e-6)
        and (padded[:, 3:] == 0).all()
    ), 'QSA_PADDED_ISOLATION'
    packed.sum().backward()
    assert (
        x.grad is not None
        and torch.isfinite(x.grad).all()
        and m.q_proj.weight.grad is not None
    ), 'QSA_MODULE_BACKWARD'
    assert all(p.grad is None for p in m.indexer.parameters()), 'QSA_INDEXER_FROZEN'
    with expect_guard('QSA_INPUT_ANGLES'):
        m(x, angles[:, :2])
    with expect_guard('QSA_PACKED_BOUNDARIES'):
        m(x, angles, cu_seqlens=torch.tensor([0, 8]))


from contextlib import contextmanager


@contextmanager
def expect_guard(tag):
    try:
        yield
    except Exception as error:
        assert tag in str(error), f"GUARD_{tag}: wrong failure {error!r}"
    else:
        assert False, f"GUARD_{tag}: missing rejection"


def test_canonical_hash_and_owner_boundary():
    from megatron.lite.model.qwen3_8_flash_next import engram as qwen_engram

    Qwen3_8_FlashNextEngramTableConfig = qwen_engram.Qwen3_8_FlashNextEngramTableConfig

    with expect_guard('HASH_RELEASE_LAYOUT'):
        NgramEmbedding(None, heads_per_ngram=1)
    table = Qwen3_8_FlashNextEngramTableConfig(8, 2).build(
        process_group=None, device='cpu', dtype=torch.float32
    )
    with expect_guard('ROW_LOOKUP_GATHER_ROWS_REQUIRED'):
        table(torch.tensor([1]))
    seen = []

    def gather(values, ids):
        seen.append(ids.clone())
        return values[ids]

    from types import SimpleNamespace

    table.lookup = SimpleNamespace(gather_rows=gather)
    result = table(torch.tensor([1, 1]))
    result.sum().backward()
    assert (
        table.weight.grad[1].tolist() == [2.0, 2.0] and len(seen) == 1
    ), 'OWNER_SHARED_CALL_BOUNDARY'


def test_ple_packed_convolution_and_backward(monkeypatch):
    from megatron.lite.model.qwen3_8_flash_next.engram import Qwen3_8_FlashNextPLELayer

    class TinyRows(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(32, 1))

        def forward(self, ids):
            return self.weight[ids % 32]

    torch.manual_seed(38)
    embedding = NgramEmbedding(TinyRows())
    with expect_guard('PLE_BACKEND_UNSUPPORTED'):
        Qwen3_8_FlashNextPLELayer(
            embedding, hidden_size=2, hc_count=4, ple_embed_dim=16, backend=object()
        )
    m = Qwen3_8_FlashNextPLELayer(
        embedding, hidden_size=2, hc_count=4, ple_embed_dim=16, dtype=torch.float32
    )
    with torch.no_grad():
        m.conv1d.weight.fill_(0.2)
    x = torch.randn(1, 8, 8, requires_grad=True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    got = m(x, ids, cu_seqlens=torch.tensor([0, 3, 8]))
    expected = torch.cat([m(x[:, :3], ids[:, :3]), m(x[:, 3:], ids[:, 3:])], 1)
    assert torch.allclose(got, expected), 'PLE_PACKED_HASH_CONV_RESET'
    got.sum().backward()
    assert (
        torch.isfinite(x.grad).all()
        and embedding.ngram_embedding.weight.grad is not None
    ), 'PLE_TABLE_GRADIENT'

    from megatron.lite.model.qwen3_8_flash_next import cp

    normalized_parts = []

    def halo(tensor, context, *, history):
        prefix = (
            torch.cat(normalized_parts, 1)[:, -history:]
            if normalized_parts
            else tensor[:, :0]
        )
        normalized_parts.append(tensor)
        return torch.nn.functional.pad(prefix, (0, 0, history - prefix.shape[1], 0))

    monkeypatch.setattr(cp, 'qwen3_8_flash_next_cp_left_halo', halo)
    outputs = []
    for rank in range(2):
        ctx = cp.Qwen3_8_FlashNextCPContext(
            None,
            rank,
            2,
            ids,
            torch.zeros_like(ids, dtype=torch.bool),
            rank * 4,
            4,
            torch.tensor([0, 3, 8]),
        )
        outputs.append(
            m(
                x[:, rank * 4 : (rank + 1) * 4],
                ids[:, rank * 4 : (rank + 1) * 4],
                cp_context=ctx,
            )
        )
    assert torch.allclose(
        torch.cat(outputs, 1), expected, atol=1e-6
    ), 'PLE_CP_PACKED_CONV_RESET'
