from types import SimpleNamespace

import pytest
import torch


def test_tp_projection_storage_and_optimizer_scope(monkeypatch):
    from megatron.lite.model.qwen3_8_flash_next import tp

    class Column(torch.nn.Module):
        def __init__(self, din, dout, ps, **kwargs):
            super().__init__()
            assert kwargs['gather_output'] is True
            self.linear = torch.nn.Linear(din, dout // ps.tp_size, bias=False)

    monkeypatch.setattr(tp, '_column_classes', lambda: (Column, Column))
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(4, 8, bias=False)
    model.embed_tokens = torch.nn.Embedding(8, 4)
    model.layers = torch.nn.ModuleList([])
    model.mlp = torch.nn.Module()
    model.mlp.experts = torch.nn.Linear(4, 8, bias=False)
    model.mlp.experts.weight.allreduce = False
    original = sum(p.numel() for p in model.parameters())
    tp.parallelize_projections(model, SimpleNamespace(tp_size=2, tp_rank=0))
    assert model.lm_head.linear.weight.shape == (4, 4), 'TP_STORAGE'
    assert sum(p.numel() for p in model.parameters()) < original, 'TP_STORAGE'
    assert model.lm_head.linear.weight.tensor_model_parallel is True
    assert (
        model.embed_tokens.weight.tensor_model_parallel is False
    ), 'TP_OPTIMIZER_SCOPE'
    assert not model.embed_tokens.weight.sequence_parallel, 'TP_REPLICA_MUST_NOT_SUM'
    # EDP optimizer shards are disjoint even though ETP=1 weights are replicated.
    assert model.mlp.experts.weight.tensor_model_parallel, 'TP_EXPERT_NORM_SCOPE'
    assert not model.mlp.experts.weight.sequence_parallel


@pytest.mark.parametrize(
    'name',
    [
        'lm_head.linear.weight',
        'layers.0.linear_attn.in_proj.linear.weight',
        'layers.0.linear_attn.o_proj.linear.weight',
        'layers.1.self_attn.q_proj.linear.weight',
        'layers.1.self_attn.o_proj.linear.weight',
        'layers.1.mlp.shared_expert.gate_up.linear.weight',
        'layers.1.mlp.shared_expert.down.linear.weight',
    ],
)
def test_tp_checkpoint_shard_axis(name):
    from megatron.lite.model.qwen3_8_flash_next.protocol import parameter_placements
    from torch.distributed.tensor import Shard

    assert parameter_placements(name)[3] == Shard(0), 'TP_CHECKPOINT_AXIS'


def test_tp_serial_key_mapping():
    from megatron.lite.model.qwen3_8_flash_next.tp import serial_parameter_name

    assert serial_parameter_name('lm_head.linear.weight') == 'lm_head.weight'
    assert (
        serial_parameter_name('layers.1.self_attn.q_proj.linear.weight')
        == 'layers.1.self_attn.q_proj.weight'
    )
    assert (
        serial_parameter_name('layers.0.linear_attn.in_proj.linear.weight')
        == 'layers.0.linear_attn.in_proj.linear.weight'
    )


def test_replicated_expert_reduction_counts_each_token_once():
    from megatron.lite.model.qwen3_8_flash_next.tp import finalize_replicated_experts

    class Buffer:
        def __init__(self):
            self.grad_data = torch.tensor([2.0, 6.0])

        def scale_gradients(self, factor):
            self.grad_data.mul_(factor)

    expert, dense = Buffer(), Buffer()
    chunk = SimpleNamespace(expert_parallel_buffers=[expert], buffers=[dense])

    def finish():
        # EDP SUM of two TP copies of the same tokens.
        expert.grad_data.mul_(2)

    finalize_replicated_experts([chunk], finish, 2)
    assert torch.equal(
        expert.grad_data, torch.tensor([2.0, 6.0])
    ), 'TP_EXPERT_DUPLICATE_TOKENS'
    assert torch.equal(
        dense.grad_data, torch.tensor([2.0, 6.0])
    ), 'TP_DENSE_SCALE_UNCHANGED'
