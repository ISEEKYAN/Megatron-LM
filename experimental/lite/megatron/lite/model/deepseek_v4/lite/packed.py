# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opt-in, document-at-a-time V4 correctness executor (physical padded offsets)."""

from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4Model
from megatron.lite.primitive.modules.paired_stream import packed_forward
from megatron.lite.primitive.ops.packed_objective import packed_objective
from megatron.lite.runtime.contracts.loss import get_loss_context


class PackedDeepseekV4Model(DeepseekV4Model):
    packed_recompute = False

    def forward(self, input_ids=None, **kwargs):
        params = kwargs.get("packed_seq_params")
        if params is None:
            if self.packed_recompute:
                raise ValueError("Document recompute requires packed_seq_params")
            return super().forward(input_ids=input_ids, **kwargs)
        if self.ps.cp_size != 1 or self.ps.pp_size != 1:
            raise ValueError("V4 document execution requires CP=PP=1")
        cu = params.cu_seqlens_q_padded
        if cu is None:
            raise ValueError("V4 document execution requires explicit padded offsets")
        if (
            input_ids is None
            or input_ids.ndim not in (1, 2)
            or (input_ids.ndim == 2 and input_ids.shape[0] != 1)
        ):
            raise ValueError("V4 packed input must be a single batch row")
        if kwargs.get("labels") is not None and kwargs.get("loss_mask") is None:
            raise ValueError("V4 packed training requires an aligned loss_mask")
        tokens = {"input_ids": input_ids.reshape(1, -1)}
        if kwargs.get("position_ids") is not None:
            tokens["position_ids"] = kwargs["position_ids"].reshape(1, -1)
        routers = [
            m.router_replay
            for m in self.layers.modules()
            if getattr(m, "router_replay", None) is not None
        ]
        output = packed_forward(
            lambda values: super(PackedDeepseekV4Model, self).forward(
                **values, enable_mtp=False
            ),
            tokens,
            cu,
            axes=1,
            output_axes={"hidden_states": 0, "logits": 1},
            routers=routers,
            recompute=self.packed_recompute,
        )
        context = get_loss_context()
        labels, mask = kwargs.get("labels"), kwargs.get("loss_mask")
        output.update(
            packed_objective(
                output.pop("logits"),
                None if labels is None else labels.reshape(1, -1),
                None if mask is None else mask.reshape(1, -1),
                temperature=kwargs.get("temperature", 1.0),
                calculate_entropy=kwargs.get("calculate_entropy", False),
                loss_scale=1.0 if context is None else context.loss_scale,
                return_log_probs=context is None or context.return_log_probs,
            )
        )
        return output
