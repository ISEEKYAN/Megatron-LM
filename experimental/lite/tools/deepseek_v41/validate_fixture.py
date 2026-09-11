"""Verify generated fixture bytes and conversion independently of the generator."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def raw(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def validate(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["file_digests"].items():
        assert (
            hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
        ), name
    assert (
        hashlib.sha256((directory / "expected_cards.json").read_bytes()).hexdigest()
        == manifest["expected_cards_sha256"]
    )
    release = load_file(directory / "release.safetensors")
    converted = load_file(directory / "converted.safetensors")
    families = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "docs/contracts/deepseek_v41/weights.json"
        ).read_text()
    )["families"]
    expected = set()
    for family in families:
        pattern = family["pattern"]
        if pattern.startswith("mtp."):
            continue
        domains = list(family["indices"])
        if ".experts.{}." in pattern:
            domains[-1] = range(8)
        if pattern.startswith("vision.blocks."):
            domains[0] = range(2)
        expected.update(pattern.format(*v) for v in itertools.product(*domains))
    assert set(release) == expected and len(release) == 3204
    assert set(converted) == expected - {
        f"layers.{i}.attn.wo_a.scale" for i in range(40)
    }
    records = {r["name"]: r for r in manifest["tensors"]}
    assert len(records) == len(manifest["tensors"]) == len(release)
    assert set(records) == expected
    for name, tensor in release.items():
        record, data = records[name], raw(tensor)
        assert list(tensor.shape) == record["shape"], name
        assert str(tensor.dtype) == record["dtype"], name
        assert len(data) == record["byte_length"], name
        assert hashlib.sha256(data).hexdigest() == record["sha256"], name
        if name.endswith(".wo_a.scale"):
            assert record["converted_key"] is None
        elif name.endswith(".wo_a.weight"):
            # Independent E8M0 interpretation and block expansion, not loader code.
            scale = release[name[:-6] + "scale"].view(torch.uint8).to(torch.int32)
            scales = (
                torch.pow(2.0, scale - 127)
                .repeat_interleave(32, 0)
                .repeat_interleave(32, 1)
            )
            decoded = (
                tensor.float() * scales[: tensor.shape[0], : tensor.shape[1]]
            ).bfloat16()
            assert raw(decoded) == raw(converted[name]), name
        else:
            assert data == raw(converted[name]), name
    inputs = load_file(directory / "inputs.safetensors")
    assert set(inputs) == set(manifest["input_keys"])
    assert inputs["packed.cu_seqlens"].tolist() == [0, 3, 12, 141]
    assert (
        inputs["packed.positions"].tolist()
        == list(range(3)) + list(range(9)) + list(range(129)) + [-1] * 3
    )
    assert inputs["packed.valid_mask"].tolist() == [True] * 141 + [False] * 3
    assert inputs["images.patches"].shape == (2, 81, 3, 14, 14)
    assert manifest["original_config"]["text_config"]["dspark_block_size"] == 5
    assert manifest["constructor_overrides"] == {"dspark_block_size": 0}
    print(
        f"FIXTURE_BYTES_OK release={len(release)} converted={len(converted)} inputs={len(inputs)}"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    validate(parser.parse_args().fixture_dir)
