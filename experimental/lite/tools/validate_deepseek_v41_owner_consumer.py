#!/usr/bin/env python3
"""Check the DeepSeek-V4.1 owner/consumer contract against official files."""
import argparse
import hashlib
import json
from pathlib import Path

from validate_deepseek_v41_ced import validate

HASHES = {
    "model": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    "config": "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
    "index": "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8",
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    [p.add_argument("--" + x, type=Path, required=True) for x in HASHES]
    a = p.parse_args()
    for n in HASHES:
        assert sha(getattr(a, n)) == HASHES[n], f"official {n} snapshot changed"
    source = a.model.read_text()
    validate(source)
    c = json.loads(a.config.read_text())["text_config"]
    r, kv, ix = (
        c["compress_ratios"],
        c["kv_source_layer_ids"],
        c["index_source_layer_ids"],
    )
    assert (
        c["num_hidden_layers"] == 40
        and len(r) == 43
        and r[40:] == [0, 0, 0]
        and kv == [2, 8, 14, 20]
        and ix == [2, 8, 14, 20, 24, 28, 32, 36]
    )
    r = r[:40]
    keys = set(json.loads(a.index.read_text())["weight_map"])
    assert len(keys) == 96085
    owners = {f"layers.{i}.attn.compressor." for i in kv} | {
        f"layers.{i}.attn.indexer." for i in ix
    }
    exact = {k for k in keys if any(k.startswith(x) for x in owners)}
    modules = {k for k in keys if ".attn.compressor." in k or ".attn.indexer." in k}
    assert exact == modules
    last_k = last_i = None
    for i, ratio in enumerate(r):
        comp = {k for k in keys if k.startswith(f"layers.{i}.attn.compressor.")}
        ind = {k for k in keys if k.startswith(f"layers.{i}.attn.indexer.")}
        assert bool(comp) == (i in kv) and bool(ind) == (i in ix)
        if comp:
            s = {k.removeprefix(f"layers.{i}.attn.compressor.") for k in comp}
            assert {"norm.weight", "wkv.weight"} <= s and (
                ratio == 1 or "wgate.weight" in s
            )
        if ind:
            s = {k.removeprefix(f"layers.{i}.attn.indexer.") for k in ind}
            assert {"wq_b.weight", "weights_proj.weight"} <= s
            assert ({"wk.weight", "k_norm.weight"} <= s) == (i in kv)
        if ratio:
            if i in kv:
                last_k = i
            if i in ix:
                last_i = i
            assert (
                last_k is not None and last_i is not None and (i < 20 or last_k == 20)
            )
    print(f"ok: 40 layers; {len(keys)} index keys; {len(exact)} exact owner keys")


if __name__ == "__main__":
    main()
