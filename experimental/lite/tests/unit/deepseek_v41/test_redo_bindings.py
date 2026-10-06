# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from dataclasses import replace

import pytest
import torch
from test_redo_parity import release_config


@pytest.mark.parametrize('corruption', [None, 'duplicate', 'missing', 'foreign'])
@pytest.mark.parametrize('trainable_engram', [False, True])
def test_parameter_bindings_require_exact_owner_inventory(
    v41_core_te, corruption, trainable_engram
):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.ckpt import binding_records

    model = protocol.build_model(
        release_config(),
        impl_cfg=protocol.ImplConfig(
            device='cpu',
            dtype=torch.float32,
            quantized=False,
            token_map=list(range(64)),
            trainable_engram=trainable_engram,
        ),
    ).chunks[0]
    model.vision.requires_grad_(False)  # Frozen parameters still need bindings.
    bindings = list(model.parameter_bindings())
    wanted = next(b for b in bindings if b.tensor is model.head.weight)
    if corruption == 'duplicate':
        bindings.append(replace(wanted, release_key='alias'))
    elif corruption == 'missing':
        bindings.remove(wanted)
    elif corruption == 'foreign':
        foreign = torch.nn.Module()
        foreign.weight = torch.nn.Parameter(wanted.tensor.detach().clone())
        assert (
            torch.equal(foreign.weight, wanted.tensor)
            and foreign.weight is not wanted.tensor
        )
        bindings[bindings.index(wanted)] = replace(wanted, owner=foreign)
    model.parameter_bindings = lambda: iter(bindings)
    if corruption is None:
        binding_records.validate_parameter_bindings(model)
    else:
        with pytest.raises(
            ValueError, match='Every parameter must have exactly one binding'
        ):
            binding_records.validate_parameter_bindings(model)
