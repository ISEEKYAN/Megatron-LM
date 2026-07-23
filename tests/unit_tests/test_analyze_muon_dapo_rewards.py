import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "docs/runs/analyze_muon_dapo_rewards.py"
    spec = importlib.util.spec_from_file_location("analyze_muon_dapo_rewards", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reward_curve_uses_verl_file_logger_shape(tmp_path):
    path = tmp_path / "arm.jsonl"
    rows = [
        {"step": 1, "data": {"critic/rewards/mean": 0.25}},
        {"step": 2, "data": {"critic/rewards/mean": 0.375}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert _module().reward_curve(path) == [(1, 0.25), (2, 0.375)]


def test_svg_contains_both_named_curves():
    rendered = _module().svg({"muon": [(1, 0.2), (2, 0.4)], "adam": [(1, 0.2), (2, 0.3)]})
    assert "muon" in rendered
    assert "adam" in rendered
    assert rendered.count("<polyline") == 2
