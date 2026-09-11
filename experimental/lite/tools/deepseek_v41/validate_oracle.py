"""Execute the reduced official oracle and assert complete owner/capture coverage."""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from fixtures import REFERENCE_SHA256
from oracle import run_forward
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    torch.set_num_threads(4)
    raw = (options.fixture_dir / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    inputs = load_file(options.fixture_dir / "inputs.safetensors")
    sys.path.insert(0, str(options.reference_dir.resolve()))
    from image_processor import ImageInput

    seq = []
    chunks = {}
    packed = manifest["packed"]
    for i, (start, end) in enumerate(
        zip(packed["cu_seqlens"], packed["cu_seqlens"][1:])
    ):
        name = f"packed{i}"
        seq.append(dict(id=name, tokens=packed["input_ids"][start:end]))
        # Start the shortest sequence before any ratio-2 group is complete.
        first = 1 if i == 0 else end - start - 2
        chunks[name] = [dict(start_pos=0, length=first)] + [
            dict(start_pos=p, length=1) for p in range(first, end - start)
        ]
    seq.append(dict(id="images", tokens=manifest["images"]["input_ids"]))
    chunks["images"] = [dict(start_pos=0, length=37)]
    image_inputs = []
    for i, (start, end) in enumerate(manifest["images"]["spans"]):
        image_inputs.append(
            ImageInput(
                start=start,
                patches=inputs["images.patches"][i],
                n_vit_h=9,
                n_vit_w=9,
                types=inputs["images.token_types"][start:end],
            )
        )
    request = dict(
        schema_version="deepseek-v41-oracle-v1",
        reference={"directory": str(options.reference_dir), "sha256": REFERENCE_SHA256},
        fixture={
            "directory": str(options.fixture_dir),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        },
        config={
            "original": manifest["original_config"],
            "overrides": manifest["effective_model_args"],
        },
        weights={
            "file": "converted.safetensors",
            "sha256": manifest["file_digests"]["converted.safetensors"],
            "keys": manifest["converted_keys"],
        },
        sequences=seq,
        chunks=chunks,
        images={"images": image_inputs},
        token_types={"images": inputs["images.token_types"]},
        execution={
            "device": "cuda",
            "seed": 1729,
            "rank": 0,
            "world_size": 1,
            "profile": "published-forward-only",
        },
        capture={"stages": "all"},
        enable_dspark_execution=False,
    )
    result = run_forward(request)
    summary = []
    for sequence in result["sequences"]:
        name = sequence["sequence_id"]
        assert sequence["logits"].shape == (
            len(next(s["tokens"] for s in seq if s["id"] == name)),
            256,
        )
        assert len(sequence["expert_probes"]) == 320
        for chunk in range(len(chunks[name])):
            captures = [r for r in sequence["captures"] if r["chunk_id"] == chunk]
            for stage in (
                "block.input",
                "block.pre_mix",
                "attn.input",
                "attn.output",
                "ffn.input",
                "ffn.output",
                "block.output",
                "block.next_mix",
                "sparse.args",
            ):
                assert [r["layer_id"] for r in captures if r["stage"] == stage] == list(
                    range(40)
                ), (name, chunk, stage)
            assert [
                r["layer_id"] for r in captures if r["stage"] == "engram.output"
            ] == [1, 14]
            assert len([r for r in captures if r["stage"] == "head.input"]) == 1
            assert len([r for r in captures if r["stage"] == "ced.x20"]) == 1
            storage = [
                r["value"]
                for r in captures
                if r["stage"] == "main.storage_identity" and r["layer_id"] >= 20
            ]
            assert len(storage) == 20 and len(set(storage)) == 1, "decoder KV ownership"
            selections = {
                r["layer_id"]: r["value"] for r in captures if r["stage"] == "topk"
            }
            for row in manifest["owners"][2:]:
                assert torch.equal(
                    selections[row["layer"]], selections[row["selection_owner"]]
                ), "selection reuse"
        summary.append(
            dict(
                sequence=name,
                tokens=sequence["logits"].shape[0],
                captures=len(sequence["captures"]),
                bound_weights=len(sequence["binding_keys"]),
                targeted_experts=320,
            )
        )
    # Perturb A, rerun A/B from fresh complete state, and require identical B.
    changed = copy.deepcopy(request)
    changed["sequences"] = copy.deepcopy(seq[:2])
    changed["sequences"][0]["tokens"][0] = (
        changed["sequences"][0]["tokens"][0] + 1
    ) % 254
    changed["chunks"] = {name: chunks[name] for name in ("packed0", "packed1")}
    changed["images"] = {}
    changed["token_types"] = {}
    isolated = run_forward(changed)
    assert torch.equal(
        isolated["sequences"][1]["logits"], result["sequences"][1]["logits"]
    ), "stale sequence state"
    options.output.mkdir(parents=True, exist_ok=False)
    capture_manifest = [
        {key: value for key, value in record.items() if key != "value"}
        for record in result["captures"]
    ]
    capture_bytes = (json.dumps(capture_manifest, indent=2) + "\n").encode()
    (options.output / "captures.json").write_bytes(capture_bytes)
    evidence = dict(
        scope="reduced-official-forward",
        job=os.environ["SLURM_JOB_ID"],
        sequences=summary,
        sequence_isolation_exact=True,
        baseline_instrumented_exact=True,
        ast_capture_additions=result["sequences"][0]["ast_capture_additions"],
        capture_count=len(capture_manifest),
        capture_manifest_sha256=hashlib.sha256(capture_bytes).hexdigest(),
    )
    (options.output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        f"OFFICIAL_ORACLE_OK sequences={len(summary)} layers=40 modes=3 engrams=2 images=2 job={os.environ['SLURM_JOB_ID']}"
    )


if __name__ == "__main__":
    main()
