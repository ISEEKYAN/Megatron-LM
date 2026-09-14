import pytest
import torch


def test_fla_l2norm_fixed_policy_is_serializable():
    import json

    from megatron.lite.primitive.ops import fla_l2norm

    policy = fla_l2norm.kernel_policy()
    assert json.loads(json.dumps(policy)) == policy
    assert policy == {"version": 1, "BT": 32, "num_warps": 4, "num_stages": 3}


@pytest.mark.parametrize("width", [128, 1024])
def test_fla_l2norm_launches_original_kernels_with_fixed_options(monkeypatch, width):
    from megatron.lite.primitive.ops import fla_l2norm as op

    calls = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(**kw):
                calls.append((self.name, grid, kw))
                if "x" in kw:
                    kw["y"].copy_(kw["x"])
                    kw["rstd"].fill_(1)
                else:
                    kw["dx"].copy_(kw["dy"])

            return launch

    for name in ["_FWD", "_BWD", "_FWD_LARGE", "_BWD_LARGE"]:
        monkeypatch.setattr(op, name, Kernel(name))
    x = torch.randn(2, width, requires_grad=True)
    y = op.fixed_l2norm(x)
    y.sum().backward()
    assert torch.equal(y, x)
    assert torch.equal(x.grad, torch.ones_like(x))
    assert [r[0] for r in calls] == (
        ["_FWD", "_BWD"] if width <= 512 else ["_FWD_LARGE", "_BWD_LARGE"]
    )
    for _, _, kw in calls:
        assert (kw["num_warps"], kw["num_stages"]) == (4, 3)
        if width <= 512:
            assert kw["BT"] == 32


def test_actual_fla_kernel_configs_must_match_across_ranks():
    from megatron.lite.primitive.ops.fla_l2norm import assert_kernel_configs_match

    def record(w):
        return {
            "name": "l2norm_fwd_kernel",
            "shape": [32, 128],
            "dtype": "torch.bfloat16",
            "config": {"BT": 32, "num_warps": w, "num_stages": 3},
        }

    assert_kernel_configs_match([[record(4)], [record(4)]])
    with pytest.raises(
        AssertionError, match="FLA_AUTOTUNE_WINNER_MUST_MATCH_ACROSS_RANKS"
    ):
        assert_kernel_configs_match([[record(4)], [record(8)]])
    with pytest.raises(AssertionError, match="FLA_L2NORM_PINNED_CONFIG_REQUIRED"):
        assert_kernel_configs_match([[record(8)], [record(8)]])
