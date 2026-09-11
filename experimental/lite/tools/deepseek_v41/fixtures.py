"""Deterministic reduced fixture metadata; never a release-weight substitute."""

import hashlib
import json
import math

import torch


KV_OWNERS = (2, 8, 14, 20)
INDEX_OWNERS = (2, 8, 14, 20, 24, 28, 32, 36)
REFERENCE_SHA256 = {
    "model.py":"4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    "kernel.py":"1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
    "engram.py":"11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    "vision.py":"5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c",
    "image_processor.py":"482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272",
    "inference_config.json":"2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809",
    "config.json":"8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
}


def validate_reference(directory):
    from pathlib import Path
    for name, expected in REFERENCE_SHA256.items():
        if hashlib.sha256((Path(directory)/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"pinned reference digest mismatch: {name}")


def ownership_rows():
    rows = []
    for layer in range(40):
        kv = max((owner for owner in KV_OWNERS if owner <= layer), default=None)
        index = max((owner for owner in INDEX_OWNERS if owner <= layer), default=None)
        mode = "SWA" if layer < 2 else "Full" if layer in KV_OWNERS else "Reindex" if layer in INDEX_OWNERS else "Reuse"
        rows.append(dict(layer=layer, ratio=0 if layer < 2 else 2 if layer < 20 else 1,
                         mode=mode, main_kv_owner=kv, index_k_owner=kv,
                         selection_owner=index, local_kv_owner=layer,
                         candidate_owner=20 if layer >= 20 else None,
                         engram=layer in (1,14)))
    return rows


def engram_layout():
    """Independent trial-division prime construction, without sympy or official code."""
    used, layers = set(), []
    for _ in (1,14):
        primes = []
        for _ in range(3):
            current = 30
            for _ in range(8):
                current += 1
                while current in used or any(current % divisor == 0 for divisor in range(2, math.isqrt(current)+1)):
                    current += 1
                used.add(current)
                primes.append(current)
        offsets = [sum(primes[:i]) for i in range(24)]
        layers.append(dict(primes=primes, offsets=offsets, rows=sum(primes)))
    return layers


def reduced_overrides():
    return dict(max_batch_size=1, max_seq_len=520, temperature=0., dtype="fp8", expert_dtype="fp4",
                n_layers=40, hc_mult=4, dim=128, vocab_size=256, n_heads=8, head_dim=64,
                rope_head_dim=64, q_lora_rank=64, o_groups=8, o_lora_rank=32,
                index_n_heads=4, index_head_dim=64, n_routed_experts=8, n_activated_experts=6,
                n_shared_experts=1, moe_inter_dim=64, window_size=128, index_topk=512,
                candidate_block_size=8, candidate_topk_blocks=2048, candidate_source_layer=20,
                kv_source_layers=list(KV_OWNERS), index_source_layers=list(INDEX_OWNERS),
                compress_ratios=[0,0]+[2]*18+[1]*20+[0]*3,
                engram_layer_ids=[1,14], engram_max_ngram_size=4, engram_n_heads=8,
                engram_head_dim=32, engram_vocab_size=31, engram_compressed_vocab_size=256,
                engram_num_embeddings=[layer["rows"] for layer in engram_layout()], engram_pad_id=2,
                vision_n_layers=2, vision_dim=64, vision_n_heads=4, vision_inter_dim=128,
                vision_patch_size=14, vision_downsample_ratio=3, image_token_id=255,
                norm_eps=1e-20, hc_eps=1e-6, hc_sinkhorn_iters=20, rope_theta=10000.,
                compress_rope_theta=160000., rope_factor=16, original_seq_len=65536,
                beta_fast=32, beta_slow=1)


def dense_values(shape, ordinal, role="weight"):
    """Exact rational recipe evaluated without framework RNG."""
    flat = torch.arange(math.prod(shape), dtype=torch.int64)
    values = (((17*ordinal+13*flat+7) % 101)-50).to(torch.float32) / 256
    if role in ("norm", "multiplier"):
        values = 1 + values / 16
    elif role in ("bias", "sink"):
        values = values / 16
    elif role != "weight":
        raise ValueError(f"unknown fixture role: {role}")
    return values.reshape(shape)


def packed_case():
    lengths = [3,9,129]
    offsets = [0,3,12,141]
    tokens = [(7*sample+13*position+3) % 254 for sample, length in enumerate(lengths) for position in range(length)]
    return dict(lengths=lengths, cu_seqlens=offsets, input_ids=tokens+[2]*3,
                sample_ids=[sample for sample,length in enumerate(lengths) for _ in range(length)]+[-1]*3,
                positions=[p for length in lengths for p in range(length)]+[-1]*3,
                valid_mask=[True]*141+[False]*3)


def image_case():
    types = [-1]*37
    span = [0]+([1]*3+[2])*3+[3]
    for start in (3,19):
        types[start:start+14] = span
    tokens = [255 if kind >= 0 else (13*position+3) % 254 for position,kind in enumerate(types)]
    return dict(input_ids=tokens, token_types=types, text_mask=[kind < 0 for kind in types],
                spans=[[3,17],[19,33]], patch_grids=[[9,9],[9,9]], patch_shape=[81,3,14,14])


def metadata():
    return dict(schema_version=1, profile="reduced-forward-v1", reduced=True,
                overrides=reduced_overrides(), owners=ownership_rows(), engram_layout=engram_layout(),
                weight_recipe="rational-v1", trainability="not_applicable",
                active_decisions=[], inactive_decisions={"O08":"no table update", "O09":"no master weights", "O12":"no indexer objective"},
                packed=packed_case(), images=image_case(),
                operator_lengths=[1,2,3,7,8,9,127,128,129,511,512,513,16383,16384,16385],
                rope_positions=[0,1,127,128,65535,65536,131071,1048575])


def metadata_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class IdentityTokenizer:
    """Synthetic reduced tokenizer with 256 distinct normalized token strings."""
    def __init__(self):
        self.backend_tokenizer = self

    def __len__(self):
        return 256

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.id_to_token(i) for i in ids)

    def id_to_token(self, index):
        return f"token{index:03d}"


def generate(reference_dir, output):
    """Generate complete reduced release/converted weights from official shapes.

    Import dependencies are real and required. Call this in a fresh process;
    no kernel or tokenizer-normalization substitutes are installed.
    """
    import dataclasses
    import importlib
    import itertools
    import sys
    from pathlib import Path

    from safetensors.torch import save_file
    from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8, dequantize_block_fp8
    from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

    reference_dir, output = Path(reference_dir), Path(output)
    validate_reference(reference_dir)
    sys.path.insert(0, str(reference_dir.resolve()))
    official = importlib.import_module("model")
    source_config = json.loads((reference_dir / "inference_config.json").read_text())
    args_map = dataclasses.asdict(official.ModelArgs())
    unknown = source_config.keys() - args_map.keys()
    if unknown:
        raise ValueError(f"unknown official ModelArgs fields: {sorted(unknown)}")
    args_map.update(source_config)
    args_map.update(reduced_overrides())
    args_map["dspark_block_size"] = 0  # recorded constructor-only waiver, never exported config
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cpu"):
            model = official.Transformer(official.ModelArgs(**args_map), tokenizer=IdentityTokenizer())
    finally:
        torch.set_default_dtype(old_dtype)
    state = model.state_dict()
    names = set(state)
    names.update(name[:-6]+"scale" for name in state if name.endswith(".wo_a.weight"))
    root = Path(__file__).resolve().parents[2]
    families = json.loads((root / "docs/contracts/deepseek_v41/weights.json").read_text())["families"]
    expected = set()
    for family in families:
        pattern = family["pattern"]
        if pattern.startswith("mtp."):
            continue
        domains = [list(domain) for domain in family["indices"]]
        if ".experts.{}." in pattern:
            domains[-1] = list(range(8))
        if pattern.startswith("vision.blocks."):
            domains[0] = list(range(2))
        expected.update(pattern.format(*indices) for indices in itertools.product(*domains))
    if names != expected:
        raise ValueError(f"reduced A2 names mismatch: missing={sorted(expected-names)} extra={sorted(names-expected)}")
    release, converted, records = {}, {}, []
    for ordinal, name in enumerate(sorted(names)):
        if name.endswith(".scale"):
            continue
        target = state[name]
        shape = list(target.shape)
        fp4 = target.dtype == torch.float4_e2m1fn_x2
        if fp4:
            shape[-1] *= 2
        role = "weight"
        if "norm" in name or name.endswith((".q_weight", ".k_weight")):
            role = "norm"
        elif name.endswith("_scale"):
            role = "multiplier"
        elif name.endswith(("bias", "bias_vl", "_base")):
            role = "bias"
        elif name.endswith("attn_sink"):
            role = "sink"
        dense = dense_values(shape, ordinal, role)
        scale_name = name[:-6]+"scale"
        if fp4:
            value, scale = quantize_mxfp4(dense)
            release[name], release[scale_name] = value, scale
            converted[name], converted[scale_name] = value, scale
        elif target.dtype == torch.float8_e4m3fn or name.endswith(".wo_a.weight"):
            block = (1,32) if name.endswith(".engram.embed.weight") else (32,32)
            value, scale = quantize_block_fp8(dense, block, scale_format="e8m0")
            release[name], release[scale_name] = value, scale
            if name.endswith(".wo_a.weight"):
                converted[name] = dequantize_block_fp8(value, scale, block).to(target.dtype)
            else:
                converted[name], converted[scale_name] = value, scale
        else:
            release[name] = converted[name] = dense.to(target.dtype)
    assert set(release) == names and set(converted) == set(state)
    output.mkdir(parents=True, exist_ok=False)
    save_file(release, output / "release.safetensors")
    save_file(converted, output / "converted.safetensors")
    packed, images = packed_case(), image_case()
    inputs = {
        "packed.input_ids":torch.tensor(packed["input_ids"],dtype=torch.int64),
        "packed.positions":torch.tensor(packed["positions"],dtype=torch.int64),
        "packed.cu_seqlens":torch.tensor(packed["cu_seqlens"],dtype=torch.int64),
        "packed.valid_mask":torch.tensor(packed["valid_mask"],dtype=torch.bool),
        "images.input_ids":torch.tensor(images["input_ids"],dtype=torch.int64),
        "images.token_types":torch.tensor(images["token_types"],dtype=torch.int64),
        "images.patches":dense_values((2,81,3,14,14),10000).bfloat16(),
    }
    for length in metadata()["operator_lengths"]:
        if length <= 513:
            inputs[f"length{length}.input_ids"] = (torch.arange(length,dtype=torch.int64)*13+3)%254
    save_file(inputs, output / "inputs.safetensors")
    cards = (root / "docs/specs/deepseek_v41_fixture_vectors.json").read_bytes()
    (output / "expected_cards.json").write_bytes(cards)
    for name, value in sorted(release.items()):
        raw = bytes(value.reshape(-1).view(torch.uint8).tolist())
        records.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                            byte_length=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                            converted_key=None if name.endswith(".wo_a.scale") else name,
                            transform="block_fp8_to_bf16" if ".wo_a." in name else "identity_bytes"))
    manifest = metadata()
    manifest.update(effective_model_args=args_map, constructor_overrides={"dspark_block_size":0},
                    original_config=json.loads((reference_dir / "config.json").read_text()),
                    tensors=records, converted_keys=sorted(converted),
                    input_keys=sorted(inputs), expected_cards_sha256=hashlib.sha256(cards).hexdigest(),
                    file_digests={name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                                  for name in ("release.safetensors","converted.safetensors","inputs.safetensors")},
                    reference={name:hashlib.sha256((reference_dir/name).read_bytes()).hexdigest()
                               for name in ("model.py","kernel.py","engram.py","vision.py","image_processor.py","inference_config.json","config.json")})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"REDUCED_FIXTURE_GENERATED layers=40 keys={len(release)} converted={len(converted)} digest={metadata_digest(manifest)}")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    generate(options.reference_dir, options.output)
