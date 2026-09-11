"""Discriminating mutations of real D call paths, with numerical witnesses."""

import pytest
import test_attention
import test_d_reference
import test_engram
import test_hc_boundary
import test_packing
import torch
from megatron.lite.model.deepseek_v41.lite import attention, block, engram
from megatron.lite.model.deepseek_v41.lite.moe import ModalityRouter


def test_reject_ratio_one_without_compressed_rope(monkeypatch):
    original = attention.rotate

    def wrong(x, positions, config, ratio, *, inverse=False):
        return original(
            x, positions, config, 0 if ratio == 1 else ratio, inverse=inverse
        )

    monkeypatch.setattr(attention, 'rotate', wrong)
    with pytest.raises(AssertionError):
        test_d_reference.test_all_40_attention_layers_against_pinned_floating_methods()


def test_reject_reintroduced_per_head_query_rms(monkeypatch):
    original = attention.CSA2Attention.__init__

    def wrong(self, config, layer_id):
        original(self, config, layer_id)

        def normalize(module, inputs, output):
            heads = output.unflatten(-1, (config.heads, config.head_dim))
            return (
                heads * torch.rsqrt(heads.square().mean(-1, keepdim=True) + config.eps)
            ).flatten(-2)

        self.wq_b.register_forward_hook(normalize)

    monkeypatch.setattr(attention.CSA2Attention, '__init__', wrong)
    with pytest.raises(AssertionError):
        test_d_reference.test_all_40_attention_layers_against_pinned_floating_methods()


def test_reject_current_block_premix_instead_of_incoming_pair(monkeypatch):
    original = block.DeepseekV41Block._forward

    def wrong(self, hidden, pre_mix, *args):
        return original(self, hidden, self.attn_mixes(hidden)[0], *args)

    monkeypatch.setattr(block.DeepseekV41Block, '_forward', wrong)
    with pytest.raises(AssertionError):
        test_hc_boundary.test_shifted_coefficients_and_paired_recompute()


def test_reject_trainable_indexer(monkeypatch):
    original = attention.Indexer.__init__

    def wrong(self, config, owns_k):
        original(self, config, owns_k)
        self.requires_grad_(True)

    monkeypatch.setattr(attention.Indexer, '__init__', wrong)
    with pytest.raises(AssertionError):
        test_attention.test_frozen_indexers_are_excluded_from_optimizer_and_kv_still_trains()


def test_reject_bias_leaking_into_gate_weights(monkeypatch):
    original = ModalityRouter.forward

    def wrong(self, x, image_mask=None):
        weights, indices, stats = original(self, x, image_mask)
        return weights + self.bias[indices], indices, stats

    monkeypatch.setattr(ModalityRouter, 'forward', wrong)
    with pytest.raises(AssertionError):
        test_d_reference.test_modality_router_against_original_gate_and_parameter_gradients()


def test_reject_wrong_hash_layer_seed(monkeypatch):
    original = engram.hash_multipliers
    monkeypatch.setattr(
        engram,
        'hash_multipliers',
        lambda layers, order, vocab: original(
            tuple(layer + 1 for layer in layers), order, vocab
        ),
    )
    with pytest.raises(AssertionError):
        test_d_reference.test_hash_normalization_layout_and_resets_against_original_module()


def test_reject_zero_master_ste(monkeypatch):
    original = engram.EngramTable.forward

    def wrong(self, ids):
        output = original(self, ids)
        if self.master is not None:
            output = output.detach() + self.master[ids].to(output.dtype) * 0
        return output

    monkeypatch.setattr(engram.EngramTable, 'forward', wrong)
    with pytest.raises(AssertionError):
        test_engram.test_fp8_table_freeze_switch_master_gradient_and_scale_publication()


def test_reject_packed_sequences_sharing_history(monkeypatch):
    def wrong(forward, h, p, bounds, **kwargs):
        return forward(h, p, **kwargs)

    monkeypatch.setattr(test_packing, 'packed_forward', wrong)
    with pytest.raises(AssertionError):
        test_packing.test_packed_b_only_gradient_and_parameter_contributions_ignore_a()
