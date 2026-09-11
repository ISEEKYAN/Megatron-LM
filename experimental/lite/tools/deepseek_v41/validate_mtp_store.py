"""Validate complete MTP byte preservation using a supplied release checkpoint.

Run with PYTHONPATH=experimental/lite. No synthetic fallback is provided.
Source revision provenance must be established independently of this I/O check.
"""

import argparse
import itertools
import json
from pathlib import Path

from megatron.lite.model.deepseek_v41.lite.checkpoint_store import CheckpointTensorStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-ranks", type=int, default=7)
    args = parser.parse_args()
    if args.storage_ranks < 1:
        parser.error("storage-ranks must be positive")
    spec = (
        Path(__file__).resolve().parents[2] / "docs/contracts/deepseek_v41/weights.json"
    )
    families = json.loads(spec.read_text())["families"]
    expected = sorted(
        family["pattern"].format(*indices)
        for family in families
        if family["pattern"].startswith("mtp.")
        for indices in itertools.product(*family["indices"])
    )
    assert len(expected) == len(set(expected)) == 2401
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    assert {k for k in index if k.startswith("mtp.")} == set(
        expected
    ), "MTP index coverage"
    files = sorted({index[k] for k in expected})
    root = args.checkpoint.resolve()
    paths = [(root / name).resolve() for name in files]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("shard outside checkpoint directory")
    store = CheckpointTensorStore.load(paths, expected_keys=expected, key_prefix="mtp.")
    # The source index must name the actual containing shard for every key.
    for name, entry in store.entries.items():
        assert Path(entry.source_shard) == (root / index[name]).resolve(), name
    config_bytes = (root / "config.json").read_bytes()
    config = json.loads(config_bytes)
    assert config["text_config"]["dspark_block_size"] == 5
    args.output.mkdir(parents=True, exist_ok=False)
    for rank in range(args.storage_ranks):
        store.shard(rank, args.storage_ranks).save(
            args.output / f"mtp-{rank:05d}.safetensors"
        )
    restored = CheckpointTensorStore.load(
        sorted(args.output.glob("mtp-*.safetensors")), expected_keys=expected
    )
    total = 0
    for name, before in store.entries.items():
        after = restored.entries[name]
        assert (
            before.dtype,
            before.shape,
            before.byte_length,
            before.payload_digest,
        ) == (after.dtype, after.shape, after.byte_length, after.payload_digest), name
        total += before.byte_length
    (args.output / "config.json").write_bytes(config_bytes)
    evidence = dict(
        keys=len(expected),
        payload_bytes=total,
        storage_ranks=args.storage_ranks,
        tensors=restored.manifest(),
        scope="supplied-checkpoint-mtp-byte-roundtrip",
    )
    (args.output / "manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        f"MTP_BYTE_ROUNDTRIP_OK keys={len(expected)} bytes={total} ranks={args.storage_ranks}"
    )


if __name__ == "__main__":
    main()
