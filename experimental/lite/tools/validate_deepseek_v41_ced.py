"""Execute pinned official control flow with CPU probes, not production kernels."""
import ast
from types import SimpleNamespace as NS

import torch


def method(source, cls, name, namespace, stop_before=None):
    tree = ast.parse(source)
    node = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    if stop_before:
        # Execute the actual prefix; only expensive downstream scoring is excluded.
        end = next(i for i, n in enumerate(node.body)
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == stop_before for t in n.targets))
        node.body = node.body[:end]
    ns = dict(torch=torch, **namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<official>', 'exec'), ns)
    return ns[name]


def check_dataflow(source):
    shared = NS()
    def rope(x, *args):
        x.add_(7)
    def quant(x, *args, **kwargs):
        x.mul_(2)
    ns = dict(shared_attn=shared, apply_rotary_emb=rope, fp4_act_quant=quant, fp4_block_size=32)
    compress = method(source, 'Compressor', 'forward', ns)
    main = method(source, 'Attention', '_compress_kv', ns)
    index = method(source, 'Indexer', 'forward', ns, stop_before='q')
    block = method(source, 'Block', 'forward', ns)
    hc_pre = method(source, 'Block', 'hc_pre', ns)
    # Execute the official transformer loop with state-sensitive block doubles.
    loop_ns = dict(ns, make_identity_pre_mix=lambda h, n: torch.zeros(1), sample=lambda x, t: x)
    forward = method(source, 'Transformer', 'forward', loop_ns)
    class Layer:
        engram = None
        def __init__(self, i):
            self.i = i
        def __call__(self, h, start, pre, mask):
            torch.testing.assert_close(h, torch.full_like(h, float(self.i)))
            torch.testing.assert_close(pre, torch.full_like(pre, float(self.i * 2)))
            return h + 1, pre + 2
        def hc_pre(self, h, pre):
            return h.mean(2)
    model = NS(engram_hash=None, embed=lambda ids: torch.zeros(1, 3, 4), hc_mult=2,
               layers=[Layer(i) for i in range(40)], target_layer_ids=[],
               head=lambda x: x, norm=lambda x: x, temperature=1)
    forward(model, torch.zeros(1, 3, dtype=torch.long))
    window = method(source, 'Attention', '_window_kv', dict(ns, act_quant=quant,
                    fp8_block_size=128, scale_fmt=None, scale_dtype=None,
                    get_window_topk_idxs=lambda *a: None))
    seen = []
    h = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4) / 10
    p = torch.tensor([0.25, 0.75]).expand(1, 3, 2)
    compressor = NS(compress_ratio=1, wkv=lambda x: x * 3, norm=lambda x: x + 5)
    idx = NS(freqs_cis=torch.zeros(3), compress_ratio=1, rope_head_dim=4,
             owns_k=True, wk=lambda x: x * 5, k_norm=lambda x: x + 11,
             k_cache=torch.zeros(1, 3, 4))
    def topk(x, qr, latent, start, offset, length):
        if latent is not None:
            seen.append(latent.clone())
            index(idx, x, qr, latent, start, offset)
        return torch.zeros(1, 3, 1, dtype=torch.int32)
    owner = NS(compress_ratio=1, is_kv_source=True, compressor=lambda x, pos: compress(compressor, x, pos),
               compress_kv_cache=torch.zeros(1, 3, 4), freqs_cis=torch.zeros(3), rope_head_dim=4,
               _compress_topk_idxs=topk)
    expected_x = (h * p.unsqueeze(-1)).sum(2) + 13
    expected_latent = expected_x * 3 + 5
    def attention(x, start, *args):
        torch.testing.assert_close(x, expected_x)
        result, _ = main(owner, x, x, start, 0)
        torch.testing.assert_close(result, (expected_latent + 7) * 2)
        return x
    b = NS(hc_mixes=lambda *a: (p, None, None), hc_pre=lambda x, pre: hc_pre(None, x, pre),
           attn_norm=lambda x: x + 13, attn=attention, hc_post=lambda x, residual, *a: residual,
           ffn_norm=lambda x: x, ffn=lambda x, mask: x)
    for name in ('hc_attn_fn', 'hc_attn_scale', 'hc_attn_base', 'hc_ffn_fn', 'hc_ffn_scale', 'hc_ffn_base'):
        setattr(b, name, None)
    block(b, h, 0, p, None)
    torch.testing.assert_close(seen[0], expected_latent)
    torch.testing.assert_close(shared.index_k, (expected_latent * 5 + 11 + 7) * 2)
    for layer in range(21, 40):
        consumer = NS(**vars(owner))
        consumer.is_kv_source = False
        consumer.compress_kv_cache = torch.full((1, 3, 4), -999.)
        consumer.compressor = lambda *a: (_ for _ in ()).throw(AssertionError('consumer recompressed'))
        local_x = expected_x + layer
        swa = NS(window_size=128, kv_norm=lambda x: x + 17, wkv=lambda x: x * 2,
                 rope_head_dim=4, window_kv_cache=torch.zeros(1, 128, 4))
        local_kv, _ = window(swa, local_x, torch.zeros(3), 0)
        torch.testing.assert_close(local_kv, (local_x * 2 + 17 + 7) * 2)
        result, _ = main(consumer, local_x, local_x, 0, 0)
        torch.testing.assert_close(result, (expected_latent + 7) * 2)
    assert len(seen) == 1, 'only source 20 may publish latent'


def validate(source):
    check_dataflow(source)
    mutations = [
        ('h, pre_mix = layer(h, start_pos, pre_mix, image_mask)',
         'h, ignored = layer(h, start_pos, pre_mix, image_mask)'),
        ('kv = self.kv_norm(self.wkv(x))', 'kv = self.wkv(x)'),
        ('x = self.hc_pre(x, pre_mix)', 'x = x.mean(dim=2)'),
        ('x = self.attn_norm(x)', 'x = x'),
        ('self.norm(self.wkv(x))', 'self.wkv(x)'),
        ('self.k_norm(self.wk(latent))', 'self.k_norm(self.wk(x))'),
        ('if self.is_kv_source:', 'if True:'),
        ('return shared_attn.compress_kv[:bsz, :compress_len], idxs',
         'return self.compress_kv_cache[:bsz, :compress_len], idxs'),
    ]
    for old, new in mutations:
        assert old in source, 'mutation target missing'
        try:
            check_dataflow(source.replace(old, new))
        except (AssertionError, RuntimeError):
            continue
        raise AssertionError(f'wrong dataflow accepted: {new}')
    print(f'ok: official CPU dataflow probes; {len(mutations)} rejected mutations (without hash gate)')
