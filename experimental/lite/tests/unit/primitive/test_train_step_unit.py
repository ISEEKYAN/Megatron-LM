from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.lite.primitive.parallel.pipeline import forward_backward_pipelining
from megatron.lite.primitive.train_step import run_microbatch_loop


pytestmark = pytest.mark.mlite


class _ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


def _forward(model: _ScalarModel, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"weighted_tokens": model.weight * batch["tokens"]}


def _globally_normalized_loss(output, batch):
    loss = output["weighted_tokens"] / batch["global_tokens"]
    return loss, {"pg_loss": loss.detach()}


def test_external_globally_normalized_loss_sums_microbatch_gradients():
    model = _ScalarModel()
    batches = iter(
        [
            {"tokens": torch.tensor(2.0), "global_tokens": torch.tensor(8.0)},
            {"tokens": torch.tensor(6.0), "global_tokens": torch.tensor(8.0)},
        ]
    )

    output = run_microbatch_loop(
        model,
        batches,
        num_microbatches=2,
        forward_fn=_forward,
        loss_fn=_globally_normalized_loss,
        loss_is_global_batch_normalized=True,
    )

    assert model.weight.grad.item() == pytest.approx(1.0)
    assert output["loss"].item() == pytest.approx(1.0)
    assert [item["loss"] for item in output["_micro_outputs"]] == pytest.approx([0.25, 0.75])
    assert [item["metrics"]["pg_loss"].item() for item in output["_micro_outputs"]] == pytest.approx(
        [0.25, 0.75]
    )


def test_external_loss_keeps_microbatch_mean_as_default():
    model = _ScalarModel()
    batches = iter(
        [
            {"tokens": torch.tensor(2.0), "global_tokens": torch.tensor(8.0)},
            {"tokens": torch.tensor(6.0), "global_tokens": torch.tensor(8.0)},
        ]
    )

    output = run_microbatch_loop(
        model,
        batches,
        num_microbatches=2,
        forward_fn=_forward,
        loss_fn=_globally_normalized_loss,
    )

    assert model.weight.grad.item() == pytest.approx(0.5)
    assert output["loss"].item() == pytest.approx(0.5)


def test_pipeline_contract_sums_globally_normalized_external_loss():
    model = _ScalarModel()
    batches = iter(
        [
            {"tokens": torch.tensor(2.0), "global_tokens": torch.tensor(8.0)},
            {"tokens": torch.tensor(6.0), "global_tokens": torch.tensor(8.0)},
        ]
    )
    parallel_state = SimpleNamespace(pp_size=1, dp_size=1)
    config = SimpleNamespace(num_microbatches=2)

    outputs = forward_backward_pipelining(
        _forward,
        [model],
        batches,
        config,
        parallel_state,
        loss_fn=_globally_normalized_loss,
        loss_is_global_batch_normalized=True,
    )

    assert model.weight.grad.item() == pytest.approx(1.0)
    assert [item["loss"] for item in outputs] == pytest.approx([0.25, 0.75])
    assert [item["metrics"]["pg_loss"].item() for item in outputs] == pytest.approx([0.25, 0.75])
