import ast
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[3] / "tools/deepseek_v41"
sys.path.insert(0, str(TOOLS))
from config_mapping import MAPPING, WAIVERS, leaves, map_release_config
from fixtures import REFERENCE_SHA256
from oracle import forward_all_tokens, snapshot, tensor_metadata, validate_sequences

REFERENCE = Path(os.environ.get("DS41_REFERENCE_DIR", "/tmp/ds41-fixture-reference"))


def _reference_text(name):
    raw = (REFERENCE / name).read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest() == REFERENCE_SHA256[name]
    ), f"G1_REFERENCE_HASH: {name}"
    return raw.decode()


def test_external_reference_matches_pinned_source():
    bundled = (
        Path(__file__).resolve().parents[2] / "fixtures/deepseek_v41/reference/model.py"
    )
    assert (
        not bundled.exists()
    ), "G1_EXTERNAL_REFERENCE: official source must stay outside the repository"
    for name in ("model.py", "config.json", "inference_config.json"):
        assert (
            hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest()
            == REFERENCE_SHA256[name]
        )


def test_capture_clone_survives_publication_mutation():
    source = torch.tensor([56.0, 68.0])
    saved = snapshot({"latent": source})
    source.add_(7).mul_(2)
    assert saved["latent"].tolist() == [56.0, 68.0]
    assert source.tolist() == [126.0, 150.0]
    info = tensor_metadata(saved)["latent"]
    assert info["shape"] == [2] and info["dtype"] == "torch.float32"
    assert info["byte_length"] == 8
    assert info["sha256"] != tensor_metadata(source)["sha256"]


def test_all_token_head_original_method_card():
    source = REFERENCE / "model.py"
    tree = ast.parse(_reference_text("model.py"))
    head_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ParallelHead"
    )
    method = next(
        node
        for node in head_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    namespace = {"torch": torch, "F": torch.nn.functional, "world_size": 1}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )

    class Head(torch.nn.Module):
        forward = namespace["forward"]

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
            )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = Head()

        def forward(self, x):
            return None, self.head(x), None

    model = Model()
    x = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]])
    original, full = forward_all_tokens(model, x)
    assert full.tolist() == [[[1.0, 2.0, 0.0], [3.0, 5.0, 1.0], [7.0, 11.0, 3.0]]]
    assert torch.equal(original[1], full[:, -1])
    assert not model.head._forward_pre_hooks


def test_bad_sequence_schedules_fail():
    seq = [{"id": "a", "tokens": [1, 2, 3]}]
    valid = {
        "a": [
            {"start_pos": 0, "length": 1},
            {"start_pos": 1, "length": 1},
            {"start_pos": 2, "length": 1},
        ]
    }
    validate_sequences(seq, valid, 256, 520)
    for bad in (
        {"a": [{"start_pos": 1, "length": 3}]},
        {"a": [{"start_pos": 0, "length": 1}, {"start_pos": 1, "length": 2}]},
    ):
        with pytest.raises(ValueError):
            validate_sequences(seq, bad, 256, 520)
    with pytest.raises(ValueError):
        validate_sequences(seq + seq, valid, 256, 520)
    with pytest.raises(ValueError):
        validate_sequences(
            [{"id": "a", "tokens": [256]}],
            {"a": [{"start_pos": 0, "length": 1}]},
            256,
            520,
        )


def test_all_87_config_leaves_map_or_have_explicit_scope():
    import copy
    import json

    release = json.loads(_reference_text("config.json"))
    tree = ast.parse(_reference_text("model.py"))
    args = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelArgs"
    )
    defaults = {
        node.target.id: eval(
            compile(ast.Expression(node.value), "ModelArgs defaults", "eval"),
            {"__builtins__": {}},
        )
        for node in args.body
        if isinstance(node, ast.AnnAssign)
    }
    assert len(leaves(release)) == 87
    assert set(leaves(release)) == set(MAPPING) | set(WAIVERS)
    mapped = map_release_config(release, defaults)
    inference = json.loads(_reference_text("inference_config.json"))
    assert all(mapped[name] == value for name, value in inference.items())
    altered = copy.deepcopy(release)
    altered["text_config"]["hidden_size"] = 640
    assert map_release_config(altered, defaults)["dim"] == 640
    altered["unrecognized_behavior"] = 1
    with pytest.raises(ValueError, match="coverage"):
        map_release_config(altered, defaults)
