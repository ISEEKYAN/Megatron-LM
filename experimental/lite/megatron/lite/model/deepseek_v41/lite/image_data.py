# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model-local image data contract, before HC expansion and sequence packing.

Inputs are decoded PIL images and already tokenized prompts. Loading URLs and
choosing a tokenizer remain caller responsibilities. Starts are sample-local.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import ImageOps

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


@dataclass(frozen=True)
class ImageConfig:
    patch_size: int = 14
    downsample_ratio: int = 3
    max_image_tokens: int = 1024
    min_pixels: int = 295936
    max_wh_ratio: float | None = None

    def __post_init__(self):
        if (
            self.patch_size < 1
            or self.downsample_ratio < 1
            or self.max_image_tokens < 4
            or self.min_pixels < 0
            or (self.max_wh_ratio is not None and self.max_wh_ratio <= 0)
        ):
            raise ValueError('Invalid image preprocessing configuration')


@dataclass(frozen=True)
class ImageInput:
    start: int
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    types: torch.Tensor


def image_token_types(height, width):
    if min(height, width) < 1:
        raise ValueError('Image token grid must be positive')
    return torch.tensor(
        [IMAGE_START] + ([IMAGE] * width + [IMAGE_NEW_LINE]) * height + [IMAGE_END],
        dtype=torch.int64,
    )


def plan_image_grid(width, height, config=ImageConfig()):
    if min(width, height) <= 0:
        raise ValueError('Image dimensions must be positive')
    p, r = config.patch_size, config.downsample_ratio
    if config.max_wh_ratio is not None:
        width = min(width, height * config.max_wh_ratio)
    if width * height < config.min_pixels:
        scale = math.sqrt(config.min_pixels / (width * height))
        width, height = int(width * scale), int(height * scale)
    best_h, best_w = math.ceil(height / p) * p, math.ceil(width / p) * p
    grid = lambda h, w: (math.ceil((h // p) / r), math.ceil((w // p) / r))
    gh, gw = grid(best_h, best_w)
    budget = config.max_image_tokens
    if gh * (gw + 1) + 2 > budget:
        aspect = height / width
        floating_w = math.sqrt((budget - 2) / aspect + 0.25) - 0.5
        floating_h = floating_w * aspect
        cell = p * r
        if floating_w < 1:
            best_h, best_w = (budget - 2) // 2 * cell, cell
        elif floating_h < 1:
            best_h, best_w = cell, (budget - 3) * cell
        else:
            scale = min(
                math.floor(floating_w) * cell / width, math.floor(floating_h) * cell / height
            )
            best_h, best_w = (math.floor(height * scale / p) * p, math.floor(width * scale / p) * p)
        gh, gw = grid(best_h, best_w)
    if min(gh, gw) < 1 or gh * (gw + 1) + 2 > budget:
        raise ValueError('Image resize cannot satisfy token budget')
    return gh, gw, best_h, best_w


def preprocess_image(image, config=ImageConfig()):
    image = image.convert('RGB')
    gh, gw, height, width = plan_image_grid(image.width, image.height, config)
    if config.max_wh_ratio is not None and image.width >= config.max_wh_ratio * image.height:
        image = image.resize((width, height))
    else:
        image = ImageOps.pad(image, (width, height), color=(127, 127, 127))
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    pixels = ((pixels - 0.5) / 0.5).to(torch.bfloat16)
    p = config.patch_size
    nh, nw = height // p, width // p
    patches = pixels.reshape(3, nh, p, nw, p).permute(1, 3, 0, 2, 4).reshape(nh * nw, 3, p, p)
    return patches, nh, nw, gh, gw


def prepare_image_inputs(tokens, images, image_token_id, config=ImageConfig()):
    """Expand each placeholder into its full span, preserving image order."""
    if sum(token == image_token_id for token in tokens) != len(images):
        raise ValueError('Image placeholder count differs from image count')
    ids, types, spans = [], [], []
    iterator = iter(images)
    for token in tokens:
        if token != image_token_id:
            ids.append(token)
            types.append(TEXT)
            continue
        patches, nh, nw, gh, gw = preprocess_image(next(iterator), config)
        layout = image_token_types(gh, gw)
        spans.append(ImageInput(len(ids), patches, nh, nw, layout))
        ids.extend([image_token_id] * len(layout))
        types.extend(layout.tolist())
    return ids, types, spans or None


def merge_image_embeddings(tokens, images, features, image_start, image_end, image_newline):
    """Out-of-place differentiable replacement of validated [B,S,D] spans.

    ``features[b][i]`` is the aligner output for ``images[b][i]``. Call this
    before ``expand_hc``; gradients from every HC copy then reach the owner.
    """
    if tokens.ndim != 3 or len(images) != len(tokens) or len(features) != len(tokens):
        raise ValueError('Expected matching batch lists and tokens [B,S,D]')
    result = tokens.clone()
    for batch, (sample, encoded) in enumerate(zip(images, features)):
        sample, encoded = sample or (), encoded or ()
        if len(sample) != len(encoded):
            raise ValueError('Every image requires one feature tensor')
        end = 0
        for img, values in zip(sample, encoded):
            layout = img.types.to(device=tokens.device)
            size = layout.numel()
            if (
                layout.ndim != 1
                or size < 4
                or img.start < end
                or img.start + size > tokens.shape[1]
            ):
                raise ValueError('Image spans must be ordered, disjoint and in bounds')
            # Validate row delimiters as well as the feature count.
            kinds = layout.tolist()
            rows = kinds[1:-1]
            if (
                kinds[0] != IMAGE_START
                or kinds[-1] != IMAGE_END
                or not rows
                or rows[-1] != IMAGE_NEW_LINE
                or any(k not in (IMAGE, IMAGE_NEW_LINE) for k in rows)
            ):
                raise ValueError('Invalid image span delimiters')
            widths, width = [], 0
            for kind in rows:
                if kind == IMAGE:
                    width += 1
                else:
                    widths.append(width)
                    width = 0
            if not widths or min(widths) < 1 or len(set(widths)) != 1:
                raise ValueError('Image rows must have equal positive widths')
            if values.shape != (sum(widths), tokens.shape[-1]):
                raise ValueError('Aligner feature shape differs from image slots')
            span = result[batch, img.start : img.start + size]
            for kind, value in (
                (IMAGE_START, image_start),
                (IMAGE_END, image_end),
                (IMAGE_NEW_LINE, image_newline),
                (IMAGE, values),
            ):
                selected = layout == kind
                if kind != IMAGE:
                    # Expand before casting: the broadcast reduction must happen
                    # in the FP32 owner's dtype, not in the BF16 residual dtype.
                    value = value.expand(int(selected.sum()), -1)
                span[selected] = value.to(device=tokens.device, dtype=tokens.dtype)
            end = img.start + size
    return result
