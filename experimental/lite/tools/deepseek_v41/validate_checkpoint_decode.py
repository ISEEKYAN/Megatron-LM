"""Compare sampled real MTP weights with the pinned official conversion code.

CPU numerical binding only: this does not certify the dynamic FP8 GPU GEMM path.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from megatron.lite.model.deepseek_v41.lite.checkpoint import load_weight
from megatron.lite.model.deepseek_v41.lite.checkpoint_store import CheckpointTensorStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-convert", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location(
        "official_ds41_convert", args.official_convert
    )
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    expected = [name for name in index if name.startswith("mtp.")]
    paths = [args.checkpoint / name for name in sorted({index[k] for k in expected})]
    store = CheckpointTensorStore.load(paths, expected_keys=expected, key_prefix="mtp.")
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = {
        "scope": "sampled-real-mtp-cpu-decode",
        "official_convert_sha256": hashlib.sha256(
            args.official_convert.read_bytes()
        ).hexdigest(),
        "cases": [],
    }
    for stage in range(3):
        for suffix in (
            "ffn.experts.0.w1.weight",
            "attn.wkv.weight",
            "attn.kv_norm.weight",
        ):
            name = f"mtp.{stage}.{suffix}"
            with safe_open(
                args.checkpoint / index[name], framework="pt", device="cpu"
            ) as reader:
                weight = reader.get_tensor(name)
            scale_name = name[:-6] + "scale"
            if scale_name in index:
                with safe_open(
                    args.checkpoint / index[scale_name], framework="pt", device="cpu"
                ) as reader:
                    scale = reader.get_tensor(scale_name)
                if weight.dtype == torch.int8:
                    # Execute the original converter, then its FP8 block interpretation.
                    weight, scale = official.cast_e2m1fn_to_e4m3fn(weight, scale)
                rows, columns = weight.shape
                reference = (
                    weight.float().view(rows // 32, 32, columns // 32, 32)
                    * scale.float()[:, None, :, None]
                ).reshape(rows, columns)
            else:
                reference = weight.float()
            result = load_weight(store, name, output_dtype=torch.float32)
            error = (result - reference).abs().max().item()
            assert torch.equal(result, reference), (name, error)
            if result.ndim == 2:
                x = ((torch.arange(result.shape[1]) % 17) - 8).float()[None, :] / 16
                assert torch.equal(x @ result.T, x @ reference.T), name
            export_path = args.output / f"case-{len(evidence['cases'])}.safetensors"
            save_file({name: result.bfloat16()}, export_path)
            restored = CheckpointTensorStore.load([export_path], expected_keys=[name])
            assert torch.equal(load_weight(restored, name), result.bfloat16()), name
            evidence["cases"].append(
                dict(
                    name=name,
                    shape=list(result.shape),
                    max_abs_error=error,
                    bf16_roundtrip_max_abs_error=0.0,
                )
            )
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        f"REAL_MTP_CPU_DECODE_OK cases={len(evidence['cases'])} max_abs_error=0 bf16_reload_error=0"
    )


if __name__ == "__main__":
    main()
