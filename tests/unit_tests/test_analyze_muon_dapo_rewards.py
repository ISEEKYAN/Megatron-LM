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


def test_reward_curve_uses_verl_console_shape(tmp_path):
    path = tmp_path / "arm.log"
    path.write_text(
        "prefix step:1 - training/global_step:1 - critic/rewards/mean:-1.0 - suffix\n"
        "prefix step:2 - training/global_step:2 - critic/rewards/mean:-0.9375 - suffix\n"
    )
    assert _module().reward_curve(path) == [(1, -1.0), (2, -0.9375)]


def test_svg_contains_both_named_curves():
    rendered = _module().svg({"muon": [(1, 0.2), (2, 0.4)], "adam": [(1, 0.2), (2, 0.3)]})
    assert "muon" in rendered
    assert "adam" in rendered
    assert rendered.count("<polyline") == 2


def test_summarize_uses_window_gain_and_linear_slope():
    curve = [(step, float(step - 6)) for step in range(1, 11)]
    summary = _module().summarize(curve, window_size=3)
    assert summary["first_window_mean"] == -4.0
    assert summary["last_window_mean"] == 3.0
    assert summary["window_gain"] == 7.0
    assert summary["linear_slope"] == 1.0


def test_verdict_rejects_short_spike_followed_by_reward_collapse():
    curves = {
        "muon": [(step, reward) for step, reward in enumerate(
            [-1.0, -0.9375, -1.0, -0.875] + [-1.0] * 26, start=1
        )],
        "adam": [(step, reward) for step, reward in enumerate(
            [-1.0] * 10 + [-0.75] * 10 + [-0.25] * 10, start=1
        )],
    }
    summary = _module().summarize_curves(curves, window_size=10)
    assert summary["verdict"] == {
        "muon_reward_increased": False,
        "muon_gain_not_below_adam": False,
        "muon_last_window_not_below_adam": False,
        "hard_gate_passed": False,
    }


def test_verdict_accepts_sustained_muon_gain_not_below_adam():
    curves = {
        "muon": [(step, step / 10) for step in range(1, 31)],
        "adam": [(step, step / 20) for step in range(1, 31)],
    }
    summary = _module().summarize_curves(curves, window_size=10)
    assert summary["verdict"]["muon_reward_increased"]
    assert summary["verdict"]["muon_gain_not_below_adam"]
    assert summary["verdict"]["muon_last_window_not_below_adam"]
    assert summary["verdict"]["hard_gate_passed"]
