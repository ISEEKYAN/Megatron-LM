# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Frozen release ownership and per-call BF16 materialization for Core consumers."""
import torch

from megatron.lite.primitive.quantization.frozen import FrozenWeight


def install(module, prefix, model, reader):
    """Replace this newly constructed layer's large matrices before the next layer."""
    from megatron.lite.model.deepseek_v41.lite.checkpoint import _expert_sources, source_name
    from megatron.lite.model.deepseek_v41.lite.export import release_format

    for local, parameter in list(module.named_parameters()):
        name = prefix + local
        if name.endswith(".ape"):
            continue  # generated position parameter; identical to strict checkpoint loading
        keys = _expert_sources(name, model.release, model.ps)
        if keys is None:
            if name.startswith('engrams.'):
                _, layer, suffix = name.split('.', 2)
                keys = [f'layers.{layer}.engram.{suffix}']
            else:
                keys = [source_name(name)]
        kind = release_format(keys[0])
        if kind not in ('mxfp4', 'fp8_block'):
            continue
        values = []
        for key in keys:
            codes = reader._get_raw_tensor(key, parameter.device)
            scale = reader._get_raw_tensor(key[:-6] + 'scale', parameter.device)
            shape = (parameter.shape[0] // len(keys), parameter.shape[1])
            value = FrozenWeight(codes, scale, shape, kind, parameter.dtype)
            values.append(value)
        weight = values[0] if len(values) == 1 else torch.cat(values)
        rows = parameter.shape[0] // len(keys)
        for index, key in enumerate(keys):
            # Reader pieces are views of the final fused owner, not duplicate GPU payloads.
            c = weight.codes[index * rows : (index + 1) * rows]
            scale_rows = rows if kind == 'mxfp4' else rows // 32
            sf = weight.scale[index * scale_rows : (index + 1) * scale_rows]
            reader._frozen_native[key] = FrozenWeight(
                c, sf, (rows, parameter.shape[1]), kind, parameter.dtype
            )
            reader._frozen_raw[key], reader._frozen_raw[key[:-6] + 'scale'] = c, sf
        owner_name, _, attribute = local.rpartition('.')
        owner = module.get_submodule(owner_name) if owner_name else module
        replacement = torch.nn.Parameter(weight, requires_grad=False)
        for attribute_name, value in parameter.__dict__.items():
            setattr(replacement, attribute_name, value)
        setattr(owner, attribute, replacement)
        # Core DDP uses the same TE compatibility cleanup after remapping weights.
        if hasattr(owner, 'weight_tensor'):
            owner.weight_tensor = None


def prepare_reader(reader):
    """Reuse the strict Lite loader while bypassing decode for already native weights."""
    if hasattr(reader, '_frozen_native'):
        return
    reader._frozen_native = {}
    reader._frozen_raw = {}
    original_raw, original_get = reader._get_raw_tensor, reader.get_tensor

    def raw(name, device):
        value = reader._frozen_raw.get(name)
        return value.to(device=device) if value is not None else original_raw(name, device)

    def get(name, **kwargs):
        value = reader._frozen_native.get(name)
        if value is None:
            return original_get(name, **kwargs)
        reader._get_raw_tensor(name, value.device)
        reader._get_raw_tensor(name[:-6] + 'scale', value.device)
        if tuple(kwargs.get('target_shape', value.shape)) != tuple(value.shape):
            raise ValueError(f'Native release logical shape mismatch for {name}')
        return value

    prefetch, close = reader.prefetch_tensors, reader.close

    def filtered_prefetch(names):
        return prefetch(name for name in names if name not in reader._frozen_native)

    def closed():
        try:
            close()
        finally:
            reader._frozen_native.clear()
            reader._frozen_raw.clear()

    reader._get_raw_tensor, reader.get_tensor = raw, get
    reader.prefetch_tensors, reader.close = filtered_prefetch, closed


def install_bf16_hooks(module):
    """Core/TE owns BF16 kernels; temporary operands never become a base master.

    These leaf hooks run inside checkpoint replay as well as the first forward.
    Deployment forwards bypass the leaf calls and consume their native codes.
    """
    for owner in module.modules():
        names = tuple(
            n for n, p in owner.named_parameters(recurse=False) if isinstance(p, FrozenWeight)
        )
        if not names:
            continue
        stack = []

        def before(current, args, names=names, stack=stack):
            originals = {n: current._parameters[n] for n in names}
            stack.append(originals)
            try:
                weights = tuple(originals.values())
                if len(weights) == 1:
                    decoded = (weights[0].decoded(),)
                else:
                    if len({(w.kind, tuple(w.shape), w.dtype) for w in weights}) != 1:
                        raise ValueError('Grouped native weights must share shape/kind/dtype')
                    codes = torch.stack([w.codes for w in weights])
                    scales = torch.stack([w.scale for w in weights])
                    batch = FrozenWeight(
                        codes,
                        scales,
                        (len(weights), *weights[0].shape),
                        weights[0].kind,
                        weights[0].dtype,
                    )
                    decoded = batch.decoded().unbind(0)
                for (name, original), matrix in zip(originals.items(), decoded):
                    value = torch.nn.Parameter(matrix, requires_grad=False)
                    for key, item in original.__dict__.items():
                        if key not in ('codes', 'scale', 'kind', '_is_param'):
                            setattr(value, key, item)
                    current._parameters[name] = value
            except BaseException:
                current._parameters.update(stack.pop())
                raise

        def after(current, args, output, stack=stack):
            if stack:
                current._parameters.update(stack.pop())
            if hasattr(current, 'weight_tensor'):
                current.weight_tensor = None

        owner.register_forward_pre_hook(before)
        owner.register_forward_hook(after, always_call=True)
