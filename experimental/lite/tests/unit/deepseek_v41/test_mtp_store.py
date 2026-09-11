"""Raw-byte storage tests; synthetic payloads are not release-weight evidence."""

import itertools
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from megatron.lite.model.deepseek_v41.lite.checkpoint_store import (
    CheckpointTensorStore,
    validate_execution,
)


def write_archive(path, tensors):
    header, payload = {}, bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        start = len(payload)
        payload.extend(raw)
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(payload)])
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


CARDS = {
    "mtp.0.packed.weight": ("I8", [2, 2], bytes.fromhex("00 7f 80 ff")),
    "mtp.0.norm.weight": ("BF16", [2], bytes.fromhex("80 3f 00 c0")),
    "mtp.0.packed.scale": ("F8_E8M0", [4], bytes.fromhex("7e 7f 80 ff")),
}


def test_byte_cards_roundtrip_and_repartition(tmp_path):
    src = tmp_path / "input.safetensors"
    write_archive(src, CARDS)
    store = CheckpointTensorStore.load([src], expected_keys=CARDS)
    original = store.manifest()
    for parts in (1, 2, 5):
        shards = [store.shard(r, parts) for r in range(parts)]
        for r, shard in enumerate(shards):
            assert list(shard.entries) == sorted(CARDS)[r::parts]
        merged = CheckpointTensorStore.merge(shards, expected_keys=CARDS)
        dst = tmp_path / f"copy{parts}.safetensors"
        merged.save(dst)
        restored = CheckpointTensorStore.load([dst], expected_keys=CARDS)
        for name, (dtype, shape, raw) in CARDS.items():
            assert restored.read(name) == raw
            assert restored.entries[name].dtype == dtype
            assert restored.entries[name].shape == tuple(shape)
            assert restored.entries[name].payload_digest == original[name]["payload_digest"]


def test_complete_synthetic_mtp_namespace(tmp_path):
    spec = Path(__file__).resolve().parents[3] / "docs/contracts/deepseek_v41/weights.json"
    families = json.loads(spec.read_text())["families"]
    names = sorted(
        family["pattern"].format(*indices)
        for family in families if family["pattern"].startswith("mtp.")
        for indices in itertools.product(*family["indices"])
    )
    assert len(names) == len(set(names)) == 2401
    tensors = {name: ("I8", [4], i.to_bytes(4, "little")) for i, name in enumerate(names)}
    src = tmp_path / "synthetic.safetensors"
    write_archive(src, tensors)
    store = CheckpointTensorStore.load([src], expected_keys=names)
    merged = CheckpointTensorStore.merge([store.shard(r, 7) for r in range(7)], expected_keys=names)
    dst = tmp_path / "repartitioned.safetensors"
    merged.save(dst)
    restored = CheckpointTensorStore.load([dst], expected_keys=names)
    assert {k: restored.read(k) for k in names} == {k: v[2] for k, v in tensors.items()}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: src.name for name in names}
    }))
    config = b'{"text_config":{"dspark_block_size":5}}\n'
    (tmp_path / "config.json").write_bytes(config)
    validator = Path(__file__).resolve().parents[3] / "tools/deepseek_v41/validate_mtp_store.py"
    result = subprocess.run([
        sys.executable, str(validator), "--checkpoint", str(tmp_path),
        "--output", str(tmp_path / "validated"), "--storage-ranks", "7",
    ], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MTP_BYTE_ROUNDTRIP_OK keys=2401 bytes=9604 ranks=7" in result.stdout
    assert (tmp_path / "validated/config.json").read_bytes() == config


def test_missing_extra_duplicate_and_backing_mutation(tmp_path):
    src = tmp_path / "source.safetensors"
    write_archive(src, CARDS)
    with pytest.raises(ValueError, match="coverage"):
        CheckpointTensorStore.load([src], expected_keys=["mtp.missing"])
    with pytest.raises(ValueError, match="duplicate"):
        CheckpointTensorStore.load([src, src], expected_keys=CARDS)
    store = CheckpointTensorStore.load([src], expected_keys=CARDS)
    with pytest.raises(ValueError, match="duplicate"):
        CheckpointTensorStore.merge([store, store], expected_keys=CARDS)
    with pytest.raises(ValueError, match="coverage"):
        CheckpointTensorStore.merge([store.shard(0, 2)], expected_keys=CARDS)
    raw = bytearray(src.read_bytes())
    raw[-1] ^= 1
    src.write_bytes(raw)
    with pytest.raises(ValueError, match="digest"):
        store.read("mtp.0.packed.scale")
    with pytest.raises(ValueError, match="digest"):
        store.save(tmp_path / "bad.safetensors")
    assert not (tmp_path / "bad.safetensors").exists()


@pytest.mark.parametrize("header,payload", [
    ({"x": dict(dtype="I8", shape=[2], data_offsets=[0, 1])}, b"x"),
    ({"x": dict(dtype="I8", shape=[-1], data_offsets=[0, 1])}, b"x"),
    ({"x": dict(dtype="UNKNOWN", shape=[1], data_offsets=[0, 1])}, b"x"),
    ({"x": dict(dtype="I8", shape=[1], data_offsets=[1, 2])}, b"xy"),
    ({"x": dict(dtype="I8", shape=[2], data_offsets=[0, 2])}, b"x"),
    ({"x": dict(dtype="I8", shape=[1], data_offsets=[0, 1])}, b"xy"),
])
def test_invalid_headers_fail(tmp_path, header, payload):
    encoded = json.dumps(header).encode()
    src = tmp_path / "invalid.safetensors"
    src.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    with pytest.raises(ValueError):
        CheckpointTensorStore.load([src], expected_keys=["x"])


def test_storage_does_not_execute_or_rewrite_config():
    config = {"text_config": {"dspark_block_size": 5}}
    validate_execution(enable_dspark_execution=False)
    assert config["text_config"]["dspark_block_size"] == 5
    with pytest.raises(NotImplementedError):
        validate_execution(enable_dspark_execution=True)
    with pytest.raises(TypeError):
        validate_execution(enable_dspark_execution=0)


def test_mixed_release_shard_and_independent_reader(tmp_path):
    from safetensors import safe_open

    src = tmp_path / "mixed.safetensors"
    write_archive(src, {"backbone.weight": ("BF16", [1], b"\x80\x3f"), **CARDS})
    store = CheckpointTensorStore.load([src], expected_keys=CARDS, key_prefix="mtp.")
    dst = tmp_path / "mtp.safetensors"
    store.save(dst)
    with safe_open(dst, framework="pt", device="cpu") as reader:
        assert set(reader.keys()) == set(CARDS)
        for name, (_, shape, raw) in CARDS.items():
            tensor = reader.get_tensor(name)
            assert list(tensor.shape) == shape
            assert bytes(tensor.view(__import__("torch").uint8).flatten().tolist()) == raw


def test_immutable_entries_and_invalid_rank(tmp_path):
    src = tmp_path / "source.safetensors"
    write_archive(src, CARDS)
    store = CheckpointTensorStore.load([src], expected_keys=CARDS)
    with pytest.raises(TypeError):
        store.entries["x"] = None
    for rank, parts in ((0, 0), (-1, 2), (2, 2), (True, 2)):
        with pytest.raises(ValueError):
            store.shard(rank, parts)
    with pytest.raises(FileExistsError):
        store.save(src)


def test_duplicate_header_key(tmp_path):
    header = b'{"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}'
    src = tmp_path / "duplicate.safetensors"
    src.write_bytes(struct.pack("<Q", len(header)) + header + b"x")
    with pytest.raises(ValueError, match="duplicate"):
        CheckpointTensorStore.load([src], expected_keys=["x"])
