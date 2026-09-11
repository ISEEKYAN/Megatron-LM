# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit post-training mask and serial external-vision backward schedule.

The model owns the parameters seen by checkpoints and optimizers. The external
copy executes vision only: weights flow owner -> copy before each microbatch;
gradients flow copy -> owner after the complete language-model backward.
"""

from copy import deepcopy
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
            raise RuntimeError('Cannot change trainability with a pending vision backward')
        model.vision_trainability = self
        for parameter in model.vision.parameters():
            parameter.requires_grad_(self.encoder)
        for parameter in model.vision.norm.parameters():
            parameter.requires_grad_(self.norm)
        for parameter in model.aligner.parameters():
            parameter.requires_grad_(self.aligner)
        for key in ('image_start', 'image_end', 'image_newline'):
            getattr(model, key).requires_grad_(self.delimiter)
        # A changed mask must never leave old gradients on frozen owners.
        for parameter in (
            *model.vision.parameters(),
            *model.aligner.parameters(),
            model.image_start,
            model.image_end,
            model.image_newline,
        ):
            if not parameter.requires_grad:
                parameter.grad = None
                if hasattr(parameter, 'main_grad'):
                    parameter.main_grad = None


class VisionSchedule:
    """One outstanding microbatch; copies are deliberately outside the module tree.

    ``backward(loss)`` is the protocol's runtime backward callback. It also
    supports external RL losses because the runtime passes the scaled loss.
    Failed or abandoned forwards must call ``abort`` before reuse.
    """

    def __init__(self, model, device):
        self.model = model
        self.vision = deepcopy(model.vision).to(device=device)
        self.aligner = deepcopy(model.aligner).to(device=device)
        self.stage = 'idle'
        self.features = []
        self.leaves = []

    def _pairs(self):
        for owner, copy in ((self.model.vision, self.vision), (self.model.aligner, self.aligner)):
            yield from zip(owner.parameters(), copy.parameters(), strict=True)

    @torch.no_grad()
    def sync_weights(self):
        if self.stage != 'idle':
            raise RuntimeError('Cannot synchronize weights with a pending vision backward')
        for owner, replica in self._pairs():
            replica.copy_(owner)
            replica.requires_grad_(owner.requires_grad)
            replica.grad = None
            if hasattr(replica, 'main_grad'):
                replica.main_grad = None
        self.vision.train(self.model.training)
        self.aligner.train(self.model.training)

    def forward(self, images):
        self.sync_weights()
        try:
            result = []
            for sample in images:
                row = []
                for img in sample or ():
                    weight = self.vision.patch_embed.proj.weight
                    patches = img.patches.to(device=weight.device, dtype=weight.dtype)
                    feature = self.aligner(
                        self.vision(patches, img.n_vit_h, img.n_vit_w), img.n_vit_h, img.n_vit_w
                    )
                    leaf = feature.detach().to(self.model.embed.weight.device)
                    leaf.requires_grad_(feature.requires_grad)
                    self.features.append(feature)
                    self.leaves.append(leaf)
                    row.append(leaf)
                result.append(row)
        except Exception:
            self.abort()
            raise
        self.stage = 'vision_forward' if torch.is_grad_enabled() else 'idle'
        if self.stage == 'idle':
            self.features, self.leaves = [], []
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
            for value, leaf in zip(self.features, self.leaves)
            if value.requires_grad and leaf.grad is not None
        ]
        if active:
            torch.autograd.backward(
                [value for value, _ in active], [grad.to(value.device) for value, grad in active]
            )
        for owner, replica in self._pairs():
            if owner.requires_grad and replica.grad is not None:
                gradient = replica.grad.to(device=owner.device, dtype=owner.dtype)
                if owner.grad is None:
                    owner.grad = gradient.clone()
                else:
                    owner.grad.add_(gradient)
                if hasattr(owner, 'main_grad'):
                    owner.main_grad = owner.grad
        self.abort()

    def state_dict(self):
        """Save a completed post-training stage, never a live autograd graph.

        Model weights, accumulated gradients and optimizer state belong to the
        caller's training checkpoint. Mid-microbatch restart requires replay.
        """
        if self.stage != 'idle':
            raise RuntimeError('Checkpoint requires a completed vision backward')
        return {'version': 1, 'trainability': vars(self.model.vision_trainability).copy()}

    def load_state_dict(self, state):
        if self.stage != 'idle':
            raise RuntimeError('Restore requires a completed vision backward')
        if set(state) != {'version', 'trainability'} or state['version'] != 1:
            raise ValueError('Unsupported vision schedule checkpoint')
        VisionTrainability(**state['trainability']).apply(self.model)
        self.sync_weights()

    def abort(self):
        """Release pending graphs; the caller must discard partial LLM gradients."""
        self.features, self.leaves = [], []
        for _, replica in self._pairs():
            replica.grad = None
        self.stage = 'idle'
