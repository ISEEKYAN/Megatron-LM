# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit post-training mask and serial external-vision backward schedule.

The model owns the parameters seen by checkpoints and optimizers. The external
copy executes vision only: weights flow owner -> copy before each microbatch;
gradients flow copy -> owner after the complete language-model backward.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VisionTrainability:
    # These are post-training choices, not a pretraining unfreeze schedule.
    encoder: bool
    norm: bool
    aligner: bool
    delimiter: bool

    def __post_init__(self):
        if any(type(value) is not bool for value in vars(self).values()):
            raise TypeError('Vision trainability fields must be explicit booleans')

    def apply(self, model):
        schedule = getattr(model, 'vision_schedule', None)
        if schedule is not None and schedule.stage != 'idle':
            raise RuntimeError(
                'Cannot change trainability with a pending vision backward'
            )
        model.vision_trainability = self
        mask = model._vision_trainability
        mask.copy_(mask.new_tensor(tuple(vars(self).values())))
        for module, enabled in (
            (model.vision, self.encoder),
            (model.vision.norm, self.norm),
            (model.aligner, self.aligner),
        ):
            module.requires_grad_(enabled)
        vectors = [
            getattr(model, key) for key in ('image_start', 'image_end', 'image_newline')
        ]
        for parameter in vectors:
            parameter.requires_grad_(self.delimiter)
        # A changed mask must never leave old gradients on frozen owners.
        for parameter in (
            *model.vision.parameters(),
            *model.aligner.parameters(),
            *vectors,
        ):
            if not parameter.requires_grad:
                parameter.grad = None
                if hasattr(parameter, 'main_grad'):
                    parameter.main_grad = None


class VisionSchedule:
    """One microbatch: external vision forward, LM backward, then vision backward.

    ``backward(loss)`` accepts runtime-scaled SFT/RL losses; abort failed forwards.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.weights = {}
        self.stage = 'idle'
        self.pending = []

    def sync_weights(self):
        if self.stage != 'idle':
            raise RuntimeError(
                'Cannot synchronize weights with a pending vision backward'
            )
        # CopyBackward returns the accumulated external gradient to its FP32 owner.
        self.weights = {
            root: {
                name: value.to(self.device, copy=True)
                for name, value in getattr(self.model, root).named_parameters()
            }
            for root in ('vision', 'aligner')
        }

    def _call(self, root, *args):
        return torch.func.functional_call(
            getattr(self.model, root), self.weights[root], args, strict=True
        )

    def forward(self, images):
        self.sync_weights()
        try:
            result = []
            for sample in images:
                row = []
                for img in sample or ():
                    weight = self.weights['vision']['patch_embed.proj.weight']
                    patches = img.patches.to(device=weight.device, dtype=weight.dtype)
                    feature = self._call(
                        'aligner',
                        self._call('vision', patches, img.n_vit_h, img.n_vit_w),
                        img.n_vit_h,
                        img.n_vit_w,
                    )
                    leaf = feature.detach().to(self.model.embed.weight.device)
                    leaf.requires_grad_(feature.requires_grad)
                    self.pending.append((feature, leaf))
                    row.append(leaf)
                result.append(row)
        except Exception:
            self.abort()
            raise
        self.stage = 'vision_forward' if torch.is_grad_enabled() else 'idle'
        if self.stage == 'idle':
            self.abort()
        return result

    def backward(self, loss):
        if self.stage != 'vision_forward':
            raise RuntimeError('LLM backward requires exactly one vision forward')
        loss.backward()
        self.stage = 'llm_backward'
        self.finish_backward()

    def finish_backward(self):
        if self.stage != 'llm_backward':
            raise RuntimeError('Vision backward must follow completed LLM backward')
        active = [
            (value, leaf.grad)
            for value, leaf in self.pending
            if value.requires_grad and leaf.grad is not None
        ]
        if active:
            torch.autograd.backward(
                [value for value, _ in active],
                [grad.to(value.device) for value, grad in active],
            )
        self.abort()

    def state_dict(self):
        """Save the completed phase/mask; the caller checkpoints model/optimizer state."""
        if self.stage != 'idle':
            raise RuntimeError('Checkpoint requires a completed vision backward')
        return {
            'version': 1,
            'trainability': vars(self.model.vision_trainability).copy(),
        }

    def load_state_dict(self, state):
        if self.stage != 'idle':
            raise RuntimeError('Restore requires a completed vision backward')
        if set(state) != {'version', 'trainability'} or state['version'] != 1:
            raise ValueError('Unsupported vision schedule checkpoint')
        VisionTrainability(**state['trainability']).apply(self.model)
        self.sync_weights()

    def abort(self):
        """Release pending graphs; the caller must discard partial LLM gradients."""
        self.pending = []
        self.weights = {}
        self.stage = 'idle'
