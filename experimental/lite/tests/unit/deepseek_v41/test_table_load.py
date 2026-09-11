import pytest
import torch
from safetensors.torch import save_file

from megatron.lite.model.deepseek_v41.lite.checkpoint import load_engram_rows
from megatron.lite.model.deepseek_v41.lite.checkpoint_store import CheckpointTensorStore


def make_store(tmp_path):
    name = 'layers.0.engram.embed.weight'
    values = (torch.arange(7 * 256) % 120).byte().reshape(7, 256).view(torch.float8_e4m3fn)
    scale = (torch.arange(7 * 8) % 8 + 123).byte().reshape(7, 8).view(torch.float8_e8m0fnu)
    path = tmp_path / 'rows.safetensors'
    save_file({name: values, name[:-6] + 'scale': scale}, path)
    return CheckpointTensorStore.load([path], expected_keys=[name, name[:-6] + 'scale']), name, values, scale


def test_stream_only_local_output_without_whole_tensor_read(tmp_path, monkeypatch):
    store, name, values, scale = make_store(tmp_path)
    monkeypatch.setattr(store, 'read', lambda *args: pytest.fail('Full tensor materialization'))
    pieces, scales = [], []
    intervals = ((0, 3), (3, 5), (5, 7), (7, 7))
    for rank, (begin, end) in enumerate(intervals):
        v, s = load_engram_rows(store, name, intervals=intervals, rank=rank, device='cpu', chunk_rows=2)
        assert v.shape == (end - begin, 256)
        assert s.shape == (end - begin, 8)
        pieces.append(v.view(torch.uint8))
        scales.append(s.view(torch.uint8))
    assert torch.equal(torch.cat(pieces), values.view(torch.uint8))
    assert torch.equal(torch.cat(scales), scale.view(torch.uint8))


@pytest.mark.parametrize('intervals', [((0, 3), (4, 7)), ((0, 4), (3, 7)), ((1, 7),), ((0, 6),)])
def test_interval_gaps_overlaps_rejected(tmp_path, intervals):
    store, name, _, _ = make_store(tmp_path)
    with pytest.raises(ValueError):
        load_engram_rows(store, name, intervals=intervals, rank=0, device='cpu')


def test_stream_checks_payload_identity_even_outside_local_shard(tmp_path):
    store, name, _, _ = make_store(tmp_path)
    entry = store.entries[name]
    with open(entry.source_shard, 'r+b') as f:
        f.seek(entry.offset + entry.byte_length - 1)
        f.write(b'\x00')
    with pytest.raises(ValueError, match='digest'):
        load_engram_rows(store, name, intervals=((0, 3), (3, 7)), rank=0, device='cpu', chunk_rows=2)


def test_loaded_provider_keeps_frozen_bytes_and_trainable_master(tmp_path):
    from megatron.lite.model.deepseek_v41.lite.engram import ShardedEngramTable
    from megatron.lite.primitive.modules.engram_lookup import RowLookup
    store, name, values, scales = make_store(tmp_path)
    for trainable in (False, True):
        table = ShardedEngramTable.from_checkpoint(store, name, RowLookup((0, 7)), device='cpu', trainable=trainable, chunk_rows=2)
        assert torch.equal(table.weight.view(torch.uint8), values.view(torch.uint8))
        assert torch.equal(table.scale.view(torch.uint8), scales.view(torch.uint8))
        assert (table.master is not None) == trainable
        if trainable:
            assert table.master.dtype == torch.float32
