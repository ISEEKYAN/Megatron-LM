"""CPU contract for the MCore ShardedTensor metadata used by the shared FC bank."""

from __future__ import annotations

import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensor
from experimental.lite.tests.smoke.primitive import test_shared_fc_bank_distopt_dcp_gpu as shared_fc_gpu


def test_unsharded_sharded_tensor_surface_has_no_slice_metadata():
    data = torch.empty((2, 32, 64), dtype=torch.bfloat16)
    tensor = ShardedTensor.from_rank_offsets("shared_fc_bank", data, replica_id=(0, 0, 0))

    assert tensor.global_shape == tensor.local_shape == (2, 32, 64)
    assert tensor.global_offset == (0, 0, 0)
    assert tensor.axis_fragmentations == (1, 1, 1)
    assert tensor.flattened_range is None


def test_dist_opt_finalization_precedes_outer_optimizer_step(monkeypatch):
    calls = []
    chunks, optimizer = [object()], object()

    def production_finalize(actual_chunks, actual_optimizer):
        assert actual_chunks is chunks
        assert actual_optimizer is optimizer
        calls.append("finalize_model_grads")

    class OuterOptimizer:
        def step(self):
            calls.append("outer_step")

    monkeypatch.setattr(shared_fc_gpu, "finalize_dist_opt_grads", production_finalize)
    shared_fc_gpu._production_finalize_grads(chunks, optimizer)()
    OuterOptimizer().step()
    assert calls == ["finalize_model_grads", "outer_step"]


def test_nonowning_dist_opt_leaf_does_not_supply_a_parameter_range():
    parameter = object()

    class NonOwner:
        pass

    class Owner:
        model_param_gbuf_map = {parameter: object()}

        @staticmethod
        def _get_model_param_range_map(actual_parameter):
            assert actual_parameter is parameter
            return {"param": slice(0, 4)}

    assert shared_fc_gpu._model_param_range_or_none(NonOwner(), parameter) is None
    assert shared_fc_gpu._model_param_range_or_none(Owner(), parameter) == {"param": slice(0, 4)}


def test_dist_opt_step_is_read_from_master_parameter_group():
    master = object()

    class Wrapped:
        optimizer = type("Optimizer", (), {"param_groups": [{"params": [master], "step": 7}]})()

    assert shared_fc_gpu._parameter_group_for_master(Wrapped(), master)["step"] == 7
