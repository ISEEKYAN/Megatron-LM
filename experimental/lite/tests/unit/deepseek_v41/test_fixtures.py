import importlib.util
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("ds41_fixture_metadata", _ROOT / "tools/deepseek_v41/fixtures.py")
fixtures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixtures)


def test_40_layer_owner_identities():
    rows = fixtures.ownership_rows()
    assert len(rows) == 40
    assert [r["layer"] for r in rows if r["mode"] == "Full"] == [2,8,14,20]
    assert [r["layer"] for r in rows if r["mode"] == "Reindex"] == [24,28,32,36]
    assert sum(r["mode"] == "Reuse" for r in rows) == 30
    assert [r["layer"] for r in rows if r["engram"]] == [1,14]
    for row in rows[20:]:
        assert row["main_kv_owner"] == row["index_k_owner"] == row["candidate_owner"] == 20
        assert row["local_kv_owner"] == row["layer"]
    assert [rows[i]["selection_owner"] for i in (20,23,24,27,28,31,32,35,36,39)] == [20,20,24,24,28,28,32,32,36,36]


def test_primes_independent_primality_check():
    # Sympy's independent primality check contrasts with generator trial division.
    from sympy import isprime

    layouts = fixtures.engram_layout()
    primes = [p for layer in layouts for p in layer["primes"]]
    assert len(primes) == len(set(primes)) == 48
    assert primes[:8] == [31,37,41,43,47,53,59,61]
    assert all(isprime(p) for p in primes)
    assert primes == [n for n in range(31, primes[-1]+1) if isprime(n)]
    for layer in layouts:
        assert layer["offsets"][0] == 0
        assert all(offset+p == following for offset,p,following in zip(layer["offsets"],layer["primes"],layer["offsets"][1:]+[layer["rows"]]))


def test_weight_recipe_hand_known_sentinels_and_reproducibility():
    values = fixtures.dense_values((2,2),0)
    assert torch.equal(values, torch.tensor([[-43,-30],[-17,-4]]).float()/256)
    assert torch.equal(fixtures.dense_values((1,),1), torch.tensor([-26/256]))
    assert fixtures.dense_values((1,),0,"norm").item() == 1-43/4096
    assert fixtures.dense_values((1,),0,"sink").item() == -43/4096
    torch.manual_seed(12345)
    assert torch.equal(values, fixtures.dense_values((2,2),0))
    assert fixtures.metadata_digest(fixtures.metadata()) == fixtures.metadata_digest(fixtures.metadata())


def test_packed_positions_and_images():
    packed, image = fixtures.packed_case(), fixtures.image_case()
    assert packed["cu_seqlens"] == [0,3,12,141]
    assert len(packed["input_ids"]) == 144
    assert sum(packed["valid_mask"]) == 141
    assert [packed["positions"][p] for p in (0,3,12)] == [0,0,0]
    assert packed["positions"][140:] == [128,-1,-1,-1]
    assert image["spans"] == [[3,17],[19,33]]
    assert image["text_mask"].count(False) == 28
    assert image["token_types"].count(1) == 18
    assert all(token == 255 for token,kind in zip(image["input_ids"],image["token_types"]) if kind >= 0)
    for start,end in image["spans"]:
        assert image["token_types"][start] == 0
        assert image["token_types"][end-1] == 3
