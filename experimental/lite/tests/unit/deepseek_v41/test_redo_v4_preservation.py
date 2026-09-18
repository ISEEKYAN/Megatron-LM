# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare compatibility commit A with the candidate on the same nv/dev core.

Run on a Slurm CUDA worker with the recorded nv/dev source first in PYTHONPATH.
A changes core ABI adaptation only; this test isolates subsequent feature edits.
"""
import os
import hashlib
import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def v4_arms(transformer_engine_import_stub, monkeypatch):
    import megatron.core.transformer.hyper_connection
    import megatron.core.fp8_utils
    import megatron.core.transformer.experimental_attention_variant.csa
    transformer_engine_import_stub()
    from megatron.lite.primitive import transformer_engine as te
    monkeypatch.setattr(te, 'RMSNorm', torch.nn.RMSNorm)
    repo = Path(__file__).resolve().parents[5]
    baseline = '4b792be1248a49127c25bd5ee00e853119a82818'
    path = 'experimental/lite/megatron/lite/primitive/modules/attention/csa.py'
    source_path = os.environ.get('CSA_COMPAT_BASELINE')
    source = (Path(source_path).read_text() if source_path else subprocess.check_output(
        ['git', 'show', baseline+':'+path], cwd=repo, text=True))
    assert hashlib.sha256(source.encode()).hexdigest() == 'ab40b8e431b412080f4f2500a5c85c192b282201e797beff8f9cd1d1cac554c7'
    original = types.ModuleType('original_csa_merge_base')
    exec(compile(source, f'{baseline}:{path}', 'exec'), original.__dict__)
    from megatron.lite.primitive.modules.attention import csa
    return original, csa


@pytest.mark.parametrize('ratio', [0, 2, 4])
def test_v4_default_forward_and_all_parameter_gradients_are_bitwise(v4_arms, ratio):
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.primitive.parallel.state import ParallelState
    original, current = v4_arms
    config = DeepseekV4Config(hidden_size=32, num_attention_heads=2, head_dim=32,
        qk_rope_head_dim=4, q_lora_rank=32, o_lora_rank=8, o_groups=1,
        index_n_heads=2, index_head_dim=32, index_topk=2, sliding_window=4,
        compress_ratios=[ratio], num_hidden_layers=1)
    kwargs = dict(layer_idx=0, ps=ParallelState(), apply_dsa_kernel_fusion=False)
    torch.manual_seed(120)
    assert torch.cuda.is_available(), "Run preservation on a Slurm CUDA worker"
    expected = original.CompressedSparseAttention(config, **kwargs).cuda()
    actual = current.CompressedSparseAttention(config, **kwargs).cuda()
    actual.load_state_dict(expected.state_dict(), strict=True)
    left = torch.randn(1,8,32,device="cuda",requires_grad=True)
    right = left.detach().clone().requires_grad_()
    positions = torch.arange(8,device="cuda").reshape(1,8)
    from megatron.core.packed_seq_params import PackedSeqParams
    packed = PackedSeqParams(cu_seqlens_q=torch.tensor([0,8], dtype=torch.int32,device="cuda"),
        cu_seqlens_kv=torch.tensor([0,8], dtype=torch.int32,device="cuda"), max_seqlen_q=8,
        max_seqlen_kv=8, qkv_format='thd')
    y = expected(left, position_ids=positions, packed_seq_params=packed)
    z = actual(right, position_ids=positions, packed_seq_params=packed)
    assert torch.equal(y,z)
    y.square().sum().backward()
    z.square().sum().backward()
    assert torch.equal(left.grad,right.grad)
    for (name,p), (other,q) in zip(expected.named_parameters(), actual.named_parameters(), strict=True):
        assert name == other
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            assert torch.equal(p.grad,q.grad), name
