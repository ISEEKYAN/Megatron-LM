# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Image data preprocessing and embedding replacement before sequence packing.

Inputs are decoded PIL images and already tokenized prompts. Loading URLs and
choosing a tokenizer remain caller responsibilities. Starts are sample-local.
"""

from dataclasses import dataclass

import torch

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


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


def merge_image_embeddings(
    tokens, images, features, image_start, image_end, image_newline
):
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


def validate_image_spans(input_ids, images, token_types=None, cu_seqlens=None):
    """Validate image positions against packed documents and return a token mask."""
    if images is not None:
        if len(images) != len(input_ids):
            raise ValueError('Image batch size differs from input IDs')
        expected_types = torch.full_like(input_ids, TEXT)
        for batch, sample in enumerate(images):
            for img in sample or ():
                if cu_seqlens is not None:
                    boundaries = cu_seqlens.tolist()
                    if not any(
                        a <= img.start and img.start + img.types.numel() <= b
                        for a, b in zip(boundaries, boundaries[1:])
                    ):
                        raise ValueError(
                            'Image span crosses a packed sequence boundary'
                        )
                expected_types[batch, img.start : img.start + img.types.numel()] = (
                    img.types.to(input_ids.device)
                )
        if token_types is not None and not torch.equal(
            token_types.to(input_ids.device), expected_types
        ):
            raise ValueError('Token types disagree with image spans')
        return expected_types >= 0
    elif token_types is not None:
        if token_types.shape != input_ids.shape or (token_types != TEXT).any():
            raise ValueError('Image token types require image inputs')
