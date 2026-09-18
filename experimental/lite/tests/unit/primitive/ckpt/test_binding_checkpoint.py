# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The binding checkpoint pipeline must work without importing a model package."""

from types import SimpleNamespace

import torch
from megatron.lite.primitive.ckpt import binding_checkpoint as checkpoint
from megatron.lite.primitive.ckpt.bindings import BoundModule, TensorBinding
from megatron.lite.primitive.ckpt.tensor_archive import CheckpointTensorStore
from safetensors.torch import save_file


class ToyModel(BoundModule):
    def __init__(self, archive):
        super().__init__()
        self.initialize_bindings(None, 1)
        self.layers = [None]
        self.weight = torch.nn.Parameter(torch.arange(6).reshape(2, 3).float())
        self._bind('toy.weight', self, 'weight', 'weight')
        self.archival_bindings = {
            'archive.weight': TensorBinding('archive.weight', self, None, 'archival')
        }
        self.archival_store = archive
        self.ps = SimpleNamespace(ep_size=1, dp_cp_size=1)
        self.config = SimpleNamespace(to_hf_dict=lambda: {'architecture': 'toy'})


def test_generic_binding_checkpoint_roundtrip(tmp_path):
    source = tmp_path / 'archive.safetensors'
    save_file({'archive.weight': torch.tensor([1.25, -0.0])}, source)
    archive = CheckpointTensorStore.load([source], expected_keys=['archive.weight'])
    spec = SimpleNamespace(
        optional_prefix='archive.',
        staging_prefix='.toy-',
        archive_required_message='Archive required',
        frozen_storage_message='Encoded table required',
        expand_bindings=lambda model, available: available,
        row_shard=lambda owner: None,
        frozen_storage=lambda owner: None,
        refresh_storage=lambda owner: None,
        row_block=lambda name: None,
    )
    model = ToyModel(archive)
    checkpoint.save_model(model, tmp_path / 'saved', spec=spec)
    loaded = ToyModel(None)
    with torch.no_grad():
        loaded.weight.zero_()
    checkpoint.load_model(loaded, tmp_path / 'saved', spec=spec)
    assert torch.equal(loaded.weight, model.weight), 'GENERIC_ACTIVE_ROUNDTRIP'
    assert loaded.archival_store.read('archive.weight') == archive.read(
        'archive.weight'
    ), 'GENERIC_ARCHIVE_BYTES'
    assert set(loaded.checkpoint_bindings) == {'toy.weight', 'archive.weight'}
