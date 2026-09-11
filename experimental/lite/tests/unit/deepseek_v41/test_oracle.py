import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[3] / "tools/deepseek_v41"
sys.path.insert(0, str(TOOLS))
from oracle import snapshot, forward_all_tokens, validate_sequences
from config_mapping import map_release_config, leaves, MAPPING, WAIVERS


def test_capture_clone_survives_publication_mutation():
    source = torch.tensor([56.0, 68.0])
    saved = snapshot({"latent": source})
    source.add_(7).mul_(2)
    assert saved["latent"].tolist() == [56.0, 68.0]
    assert source.tolist() == [126.0, 150.0]


def test_all_token_head_original_method_card():
    source = Path("/tmp/ds41-review/model.py")
    tree = ast.parse(source.read_text())
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

    release = json.loads(Path("/tmp/ds41-review/config.json").read_text())
    tree = ast.parse(Path("/tmp/ds41-review/model.py").read_text())
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
    inference = json.loads(Path("/tmp/ds41-review/inference_config.json").read_text())
    assert all(mapped[name] == value for name, value in inference.items())
    altered = copy.deepcopy(release)
    altered["text_config"]["hidden_size"] = 640
    assert map_release_config(altered, defaults)["dim"] == 640
    altered["unrecognized_behavior"] = 1
    with pytest.raises(ValueError, match="coverage"):
        map_release_config(altered, defaults)
