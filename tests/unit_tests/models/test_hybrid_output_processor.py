# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.models.hybrid.hybrid_model import HybridModel


class Decoder(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states * 2


class OutputLayer(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("Custom processor must bypass the vocabulary projection.")


def test_hybrid_forward_calls_processor_before_projection():
    model = object.__new__(HybridModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        fine_grained_activation_offloading=False,
        moe_paged_stash=False,
        moe_num_hash_layers=0,
        sequence_parallel=False,
        freeze_base_model_for_mtp=False,
        mtp_num_layers=None,
        use_mup=False,
    )
    model.pre_process = False
    model.post_process = True
    model.position_embedding_type = "none"
    model.mtp_process = False
    model.share_embeddings_and_output_weights = False
    model.decoder = Decoder()
    model.output_layer = OutputLayer()
    model.compute_language_model_loss = lambda *args: None
    hidden = torch.randn(3, 2, 4, requires_grad=True)
    context = object()
    calls = []

    def processor(**kwargs):
        calls.append(kwargs)
        return kwargs["hidden_states"].square().sum()

    output = HybridModel.forward(
        model,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        decoder_input=hidden,
        compute_mtp_loss=False,
        output_processor=processor,
        output_processor_context=context,
    )
    output.backward()
    assert calls[0]["context"] is context
    assert calls[0]["output_layer"] is model.output_layer
    assert calls[0]["config"] is model.config
    torch.testing.assert_close(hidden.grad, 8 * hidden.detach())
