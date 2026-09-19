# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch
from megatron.lite.primitive.ckpt.binding_records import TensorBinding


@pytest.mark.parametrize('corruption', [None, 'duplicate', 'missing', 'foreign'])
def test_parameter_bindings_require_exact_owner_inventory(corruption):
    from megatron.lite.primitive.ckpt import binding_records

    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)  # Frozen parameters still need bindings.
    bindings = [TensorBinding(n, model, n, 'weight') for n in ('weight', 'bias')]
    if corruption == 'duplicate':
        bindings.append(TensorBinding('alias', model, 'weight', 'weight'))
    elif corruption == 'missing':
        bindings.pop()
    elif corruption == 'foreign':
        foreign = torch.nn.Linear(3, 2)
        foreign.bias.data.copy_(model.bias)
        assert torch.equal(foreign.bias, model.bias) and foreign.bias is not model.bias
        bindings[-1] = TensorBinding('bias', foreign, 'bias', 'weight')
    model.parameter_bindings = lambda: iter(bindings)
    if corruption is None:
        binding_records.validate_parameter_bindings(model)
    else:
        with pytest.raises(
            ValueError, match='Every parameter must have exactly one binding'
        ):
            binding_records.validate_parameter_bindings(model)
