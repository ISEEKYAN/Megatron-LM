# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed next-token loss normalization, CP targets, and output reconstruction."""

import math
from dataclasses import replace

import torch
from megatron.lite.primitive.parallel.thd import roll_packed_thd_left
from torch.nn import functional as F


def prepare_microbatches(data_iter, count, *, dp_group=None, cp_rank=0, cp_size=1):
    """Sum owned token weights over the gradient averaging group (DP x CP)."""
    from megatron.lite.runtime.contracts.loss import LossContext, split_loss_context

    if count < 1:
        raise ValueError('Microbatch count must be positive')
    items = [split_loss_context(next(data_iter)) for _ in range(count)]
    total = 0.0
    for batch, _ in items:
        if batch.labels is None:
            raise ValueError('SFT normalization requires labels')
        mask = (
            torch.ones_like(batch.labels, dtype=torch.float32)
            if batch.loss_mask is None
            else batch.loss_mask
        )
        if (
            mask.shape != batch.input_ids.shape
            or not torch.isfinite(mask).all()
            or (mask < 0).any()
        ):
            raise ValueError('Expected finite nonnegative token loss weights')
        # Count exactly the weights consumed by _text_output's next-token CE.
        # The packed shift zeros each sequence tail (there is no next label),
        # so the original first-token weight does not contribute.
        shifted_mask, _ = roll_packed_thd_left(mask, cu_seqlens_padded=batch.cu_seqlens)
        if cp_size > 1:
            from megatron.lite.primitive.modules.attention.cp import (
                ContiguousCPSequence,
            )

            shifted_mask = ContiguousCPSequence(
                batch.input_ids.numel(), cp_rank, cp_size
            ).slice(shifted_mask, seq_dim=0)
        total += float(shifted_mask.sum())
    # The generic runtime divides every microbatch by count after this loss.
    dp_size = 1
    if dp_group is not None:
        tokens = torch.tensor(
            total, dtype=torch.float64, device=items[0][0].input_ids.device
        )
        torch.distributed.all_reduce(tokens, group=dp_group)
        total = float(tokens)
        dp_size = torch.distributed.get_world_size(dp_group)
    denominator = max(total, 1.0) / (count * dp_size)
    return [
        (
            batch,
            replace(context or LossContext(), normalization_denominator=denominator),
        )
        for batch, context in items
    ]


def _cp_targets(batch, cp_context):
    """Shift full documents once, then optionally select this CP rank's tokens."""
    mask = (
        torch.ones_like(batch.labels, dtype=torch.float32)
        if batch.loss_mask is None
        else batch.loss_mask
    )
    labels, mask = (
        roll_packed_thd_left(value, cu_seqlens_padded=batch.cu_seqlens)[0]
        for value in (batch.labels, mask)
    )
    denominator = mask.sum().clamp_min(1)
    if cp_context is None:
        return labels, mask, denominator
    return (
        cp_context.slice(labels, seq_dim=0),
        cp_context.slice(mask, seq_dim=0),
        denominator,
    )


def text_output(logits, batch, *, cp_context=None):
    from megatron.lite.runtime.contracts.loss import get_loss_context

    context = get_loss_context()
    temperature = 1.0 if context is None else context.temperature
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    logits = logits / temperature
    result = {'logits': logits}
    if batch.labels is not None:
        if batch.labels.shape != batch.input_ids.shape:
            raise ValueError('Labels must match packed input shape')
        if batch.loss_mask is not None and batch.loss_mask.shape != batch.labels.shape:
            raise ValueError('Loss mask must match packed input shape')
        labels, mask, denominator = _cp_targets(batch, cp_context)
        token_loss = F.cross_entropy(logits, labels, reduction='none')
        if context is not None and context.normalization_denominator is not None:
            denominator = context.normalization_denominator
            if not math.isfinite(denominator) or denominator <= 0:
                raise ValueError('Loss denominator must be finite and positive')
        result['loss'] = (token_loss * mask).sum() / denominator
        if cp_context is not None and (
            context is None or context.normalization_denominator is None
        ):
            # Prepared denominators already compensate for DP x CP averaging.
            result['loss'] = result['loss'] * cp_context.size
        if context is not None:
            result['loss'] = result['loss'] * context.loss_scale
        if context is None or context.return_log_probs:
            result['log_probs'] = -token_loss
    if context is not None and context.calculate_entropy:
        log_probs = logits.log_softmax(-1)
        result['entropy'] = -(log_probs.exp() * log_probs).sum(-1)
    return result


def unpack_forward_output(model, batch, output):
    if isinstance(output, dict):
        return {
            key: unpack_forward_output(model, batch, value)
            for key, value in output.items()
        }
    if model.ps.cp_size > 1 and isinstance(output, torch.Tensor) and output.ndim > 0:
        from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

        cp_context = ContiguousCPSequence(
            batch.total_tokens, model.ps.cp_rank, model.ps.cp_size, model.ps.cp_group
        )
        output = cp_context.gather(output, seq_dim=0)
    if (
        isinstance(output, torch.Tensor)
        and output.ndim > 0
        and output.shape[0] == batch.total_tokens
    ):
        return torch.nested.as_nested_tensor(
            list(output.split(batch.seq_lens.tolist()))
        )
    return output


from contextlib import contextmanager

from megatron.lite.primitive.modules.router_replay import (
    RouterReplay,
    RouterReplayAction,
)


@contextmanager
def _preserve_replay(instances):
    saved = [
        (r, r.router_replay_action, r.target_topk_idx, r.target_replay_mask)
        for r in instances
    ]
    try:
        yield
    finally:
        for replay, action, target, mask in saved:
            replay.router_replay_action = action
            replay.target_topk_idx, replay.target_replay_mask = target, mask


class PackedRouterReplay:
    """Bind one packed invocation to its routes before visiting logical samples.

    This adapter does not execute a model or choose a CP layout. Its offsets
    refer to the router-local token buffer, after protocol packing/CP/TP slicing.
    A recomputed *whole* packed invocation consumes one legacy FIFO entry per
    router, not one entry per sample. Checkpoints inside a sample should use
    ``router_replay_checkpoint_contexts`` to bind their own immutable targets.
    """

    def __init__(self, total_tokens):
        self.total_tokens = total_tokens
        self.entries = []
        self.next_begin = 0
        for replay in RouterReplay.global_router_replay_instances:
            action = replay.router_replay_action
            target, mask = replay.target_topk_idx, replay.target_replay_mask
            if action == RouterReplayAction.REPLAY_BACKWARD:
                if not replay.replay_backward_list:
                    raise RuntimeError('Packed replay backward target queue is empty')
                target = replay.replay_backward_list.pop(0)
                mask = replay.replay_backward_mask_list.pop(0)
            if action in (
                RouterReplayAction.REPLAY_FORWARD,
                RouterReplayAction.REPLAY_BACKWARD,
            ):
                if target is None or target.shape[0] != total_tokens:
                    raise ValueError(
                        'Packed replay target must match the local token buffer'
                    )
                if mask is not None and mask.numel() != total_tokens:
                    raise ValueError(
                        'Packed replay mask must match the local token buffer'
                    )
            self.entries.append((replay, action, target, mask, []))

    @contextmanager
    def sequence(self, begin, end):
        if begin != self.next_begin or not begin < end <= self.total_tokens:
            raise ValueError(
                'Packed replay ranges must partition the token buffer in order'
            )
        with _preserve_replay(entry[0] for entry in self.entries):
            for replay, action, target, mask, records in self.entries:
                if action == RouterReplayAction.RECORD:
                    replay.recorded_topk_idx = None
                elif action in (
                    RouterReplayAction.REPLAY_FORWARD,
                    RouterReplayAction.REPLAY_BACKWARD,
                ):
                    replay.router_replay_action = RouterReplayAction.REPLAY_FORWARD
                    replay.target_topk_idx = target[begin:end]
                    replay.target_replay_mask = (
                        None if mask is None else mask[begin:end]
                    )
            yield
            for replay, action, target, mask, records in self.entries:
                if action == RouterReplayAction.RECORD:
                    result = replay.recorded_topk_idx
                    if result is None or result.shape[0] != end - begin:
                        raise RuntimeError(
                            'Packed record did not visit every local router/token'
                        )
                    records.append(result.detach().clone())
            self.next_begin = end

    def finish(self):
        if self.next_begin != self.total_tokens:
            raise RuntimeError('Packed replay token partition is incomplete')
        for replay, action, target, mask, records in self.entries:
            if action == RouterReplayAction.RECORD:
                replay.recorded_topk_idx = torch.cat(records, dim=0)


def packed_paired_forward(
    sequence_forward,
    hidden,
    pre_mix,
    cu_seqlens,
    *,
    input_ids=None,
    image_mask=None,
    cp_context=None,
):
    """Run a pure sequence callable over each logical sample, preserving its graph.

    The callable returns (hidden, next_pre_mix), creates fresh sequence state
    per invocation, and keeps bias/statistic publication outside forward. RNG is
    consumed in sequence order, exactly as for independent calls. This is a
    correctness path; CP transport and document ownership belong to the primitive.
    """
    from contextlib import nullcontext

    from megatron.lite.primitive.utils.packed_seq import packed_sequence_ranges

    if hidden.ndim != 4 or hidden.shape[0] != 1 or pre_mix.shape != hidden.shape[:-1]:
        raise ValueError("Expected packed hidden [1,T,HC,D] and pre_mix [1,T,HC]")
    for tensor in (input_ids, image_mask):
        if tensor is not None and tensor.shape != hidden.shape[:2]:
            raise ValueError("Token inputs must match packed [1,T] dimensions")
    outputs, mixes = [], []
    replay = PackedRouterReplay(hidden.shape[1]) if cp_context is None else None
    total = hidden.shape[1] if cp_context is None else cp_context.total_length
    offset = 0
    for begin, end in packed_sequence_ranges(cu_seqlens, total):
        kwargs = {}
        if cp_context is not None:
            document = cp_context.document(begin, end)
            kwargs['cp_context'] = document
            begin, end = offset, offset + document.local_length
            offset = end
        if input_ids is not None:
            kwargs['input_ids'] = input_ids[:, begin:end]
        if image_mask is not None:
            kwargs['image_mask'] = image_mask[:, begin:end]
        with replay.sequence(begin, end) if replay is not None else nullcontext():
            h, p = sequence_forward(
                hidden[:, begin:end], pre_mix[:, begin:end], **kwargs
            )
        outputs.append(h)
        mixes.append(p)
    if replay is not None:
        replay.finish()
    return torch.cat(outputs, dim=1), torch.cat(mixes, dim=1)


from contextlib import nullcontext

from megatron.lite.primitive.parallel.route_records import router_replay_roots

_text_output = text_output


def _validate_replay(model, batch, *, model_name):
    if batch.routed_experts is not None:
        from megatron.lite.primitive.modules.router_replay import RouterReplayAction

        routers = [
            module
            for root in router_replay_roots(model, model_name=model_name)
            for module in root.modules()
            if hasattr(module, 'router_replay')
        ]
        if not routers or any(
            module.router_replay is None
            or module.router_replay.router_replay_action
            != RouterReplayAction.REPLAY_FORWARD
            for module in routers
        ):
            raise RuntimeError(
                '{model} routed inputs require an active replay driver'.replace(
                    "{model}", model_name
                )
            )


def _validate_text_batch(batch, *, multimodal=False):
    if set(batch.extras) - ({'images', 'token_types'} if multimodal else set()):
        raise NotImplementedError(
            'Text-only protocol does not accept extra modality fields'
        )
    if batch.input_ids.ndim != 1 or batch.total_tokens != batch.input_ids.numel():
        raise ValueError('Expected packed 1-D tokens matching seq_lens')
    if batch.position_ids is not None and not torch.equal(
        batch.position_ids,
        torch.cat(
            [
                torch.arange(int(n), device=batch.input_ids.device)
                for n in batch.seq_lens
            ]
        ),
    ):
        raise ValueError('Only sequence-local positions are supported')


def _forward_step(
    model, batch, *, optimizer=None, execution_model=None, model_name="Model"
):
    schedule = model.vision_schedule
    if schedule is not None and schedule.stage != 'idle':
        raise RuntimeError('Previous microbatch requires completed vision backward')
    try:
        return _forward_step_impl(
            model,
            batch,
            optimizer=optimizer,
            execution_model=execution_model,
            model_name=model_name,
        )
    except Exception:
        if schedule is not None:
            schedule.abort()
        raise


def _forward_step_impl(
    model, batch, *, optimizer=None, execution_model=None, model_name="Model"
):
    if model.ps.pp_size > 1:
        _validate_text_batch(batch)
    else:
        _validate_text_batch(batch, multimodal=True)
    _validate_replay(model, batch, model_name=model_name)
    precision = (
        torch.autocast(device_type=batch.input_ids.device.type, enabled=False)
        if hasattr(model, 'residual_dtype')
        else nullcontext()
    )
    modality = dict(batch.extras)
    cp_context = None
    ids = batch.input_ids[None]
    if model.ps.cp_size > 1:
        from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

        if (
            modality
            or batch.routed_experts is not None
            or batch.r3_replay_mask is not None
        ):
            raise NotImplementedError(
                'CP text training does not yet accept modality or replay inputs'
            )
        cp_context = ContiguousCPSequence(
            batch.total_tokens, model.ps.cp_rank, model.ps.cp_size, model.ps.cp_group
        )
        ids = cp_context.slice(ids)
    if 'token_types' in modality:
        if modality['token_types'].shape != batch.input_ids.shape:
            raise ValueError('Packed token types must match the input IDs')
        modality['token_types'] = modality['token_types'][None]
    with precision:
        output = (model if execution_model is None else execution_model)(
            ids, cu_seqlens=batch.cu_seqlens, cp_context=cp_context, **modality
        )
    result = (
        {'hidden_states': output['hidden_states']}
        if 'hidden_states' in output
        else _text_output(output['logits'][0], batch, cp_context=cp_context)
    )
    if optimizer is not None and model.training and torch.is_grad_enabled():
        optimizer.accumulate_modality_loads(output['modality_loads'])
    if model.vision_schedule is not None and model.vision_schedule.stage != 'idle':
        result['backward'] = model.vision_schedule.backward
    return result
