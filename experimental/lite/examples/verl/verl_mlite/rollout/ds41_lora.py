# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Adapter-only DS4.1 synchronization over VERL's existing bucket transport."""
import hashlib
import json
from functools import wraps

import torch


def _sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def _entries(tensors):
    result = []
    for name, value in sorted(tensors.items()):
        if not name.startswith('layers.') or '.lora_' not in name or not name.endswith('.weight'):
            raise ValueError(f'Adapter sync forbids base/buffer tensor {name}')
        if (
            value.dtype != torch.bfloat16
            or value.ndim not in (2, 3)
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f'Invalid BF16 adapter tensor {name}')
        raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        result.append(
            dict(
                name=name,
                shape=list(value.shape),
                dtype='bfloat16',
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    if not result:
        raise ValueError('Adapter sync is empty')
    return result


def _snapshot_bucket(weights):
    """Own a reused native IPC bucket with one device-to-host copy, not one per factor."""
    items = list(weights)
    if not items:
        return {}
    first = items[0][1].detach()
    same_storage = all(
        value.dtype == first.dtype
        and value.device == first.device
        and value.is_contiguous()
        and value.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
        for _, value in items
    )
    if same_storage:
        start = min(value.storage_offset() for _, value in items)
        stop = max(value.storage_offset() + value.numel() for _, value in items)
        owned = torch.as_strided(first, (stop - start,), (1,), storage_offset=start).to(
            'cpu', copy=True
        )
        return {
            name: owned[
                value.storage_offset() - start : value.storage_offset() - start + value.numel()
            ].view(value.shape)
            for name, value in items
        }
    # Non-native callers may supply separate allocations; a single concatenation
    # still avoids per-factor CUDA copies/synchronizations.
    lengths = [value.numel() for _, value in items]
    owned = torch.cat([value.detach().reshape(-1) for _, value in items]).to('cpu', copy=True)
    return {
        name: value.view(original.shape)
        for (name, original), value in zip(items, owned.split(lengths))
    }


def prepare_adapter_sync(weights, cfg, num_layers, version, base_manifest_sha256):
    from megatron.lite.model.deepseek_v41.vllm.lora import serving_targets
    from megatron.lite.primitive.modules.lora import normalize_lora_config

    cfg = normalize_lora_config(cfg)
    if (
        not cfg.enabled
        or not isinstance(version, int)
        or version < 0
        or not isinstance(base_manifest_sha256, str)
        or len(base_manifest_sha256) != 64
    ):
        raise ValueError('Adapter sync requires enabled LoRA, version and frozen base manifest')
    int(base_manifest_sha256, 16)
    tensors = {}
    for name, value in weights:
        if name in tensors:
            raise ValueError(f'Duplicate adapter tensor {name}')
        tensors[name] = value.detach().to('cpu', copy=True).contiguous()
    manifest = dict(
        policy_version=int(version),
        base_manifest_sha256=base_manifest_sha256,
        tensors=_entries(tensors),
    )
    config = dict(
        peft_type='LORA',
        r=cfg.rank,
        lora_alpha=cfg.rank if cfg.alpha is None else cfg.alpha,
        target_modules=list(serving_targets(cfg, num_layers)),
        _mlite_ds41=dict(manifest=manifest, sha256=_sha(manifest)),
    )
    return iter(sorted(tensors.items())), config


def receive_adapter(receiver, model_runner, config):
    from vllm.lora.request import LoRARequest

    tensors, terminal, errors = {}, False, []

    def consume(weights, is_last):
        nonlocal terminal
        weights = list(weights)
        if terminal:
            errors.append('Adapter bucket after terminal marker')
        for name, value in weights:
            if name in tensors:
                errors.append(f'Duplicate adapter tensor {name}')
                continue
            if '.lora_' not in name:
                errors.append(f'Base tensor in adapter publication: {name}')
                continue
            # The transport reuses its IPC buffer after this callback.
            tensors[name] = value
        # Copy before the native transport is allowed to reuse this buffer.
        tensors.update(
            _snapshot_bucket([(n, v) for n, v in weights if n in tensors and tensors[n] is v])
        )
        terminal = bool(is_last)

    receiver.receive_weights(on_bucket_received=consume)
    # Drain the existing transport before rejecting metadata: an early exception
    # would strand the sender waiting for a bucket handshake.
    if errors:
        raise ValueError('; '.join(errors))
    header = config.get('_mlite_ds41', {})
    manifest = header.get('manifest', {})
    if _sha(manifest) != header.get('sha256'):
        raise ValueError('Adapter manifest hash mismatch')
    hf = model_runner.vllm_config.model_config.hf_config
    expected_base = getattr(hf, 'mlite_base_manifest_sha256', None)
    if not expected_base or manifest['base_manifest_sha256'] != expected_base:
        raise ValueError('Adapter targets a different or undeclared frozen base')
    version = manifest['policy_version']
    if not isinstance(version, int) or version < 0:
        raise ValueError('Adapter policy version must be a nonnegative integer')
    previous = getattr(model_runner, '_mlite_ds41_adapter_ack', None)
    if previous is not None and version != previous['policy_version'] + 1:
        raise ValueError('Adapter policy versions must advance once per publication')
    if not terminal or _entries(tensors) != manifest['tensors']:
        raise ValueError('Partial or corrupt adapter publication')
    from verl.workers.rollout.vllm_rollout.utils import (
        VLLM_LORA_INT_ID,
        VLLM_LORA_NAME,
        VLLM_LORA_PATH,
    )

    # VERL generation requests always use this identity. Publication replaces
    # that id in place; policy_version and manifest hash identify the weights.
    request = LoRARequest(
        lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
    )
    peft = {k: v for k, v in config.items() if k != '_mlite_ds41'}
    model_runner.lora_manager.add_adapter_tensors(request, tensors, peft)
    torch.cuda.synchronize()  # ACK means actual copies/activation completed, not RPC dispatch.
    tensors.clear()
    ack = dict(
        policy_version=version,
        manifest_sha256=header['sha256'],
        base_manifest_sha256=expected_base,
        lora_name=request.lora_name,
        lora_int_id=request.lora_int_id,
        finished=True,
        staging_current=0,
    )
    model_runner._mlite_ds41_adapter_ack = ack
    return ack


def install():
    try:
        from verl.workers.rollout.vllm_rollout import utils
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith(('verl', 'vllm')):
            return False
        raise

    cls = getattr(utils, 'vLLMColocateWorkerExtension', None)
    if cls is None:
        return False
    original = cls.update_weights_from_ipc
    if getattr(original, '_mlite_ds41_adapter', False):
        return False

    @wraps(original)
    def update(self, peft_config=None, base_sync_done=False, use_shm=False):
        ds41 = type(self.model_runner.model).__name__.startswith('DeepseekV41')
        if not peft_config or (not ds41 and '_mlite_ds41' not in peft_config):
            return original(
                self, peft_config=peft_config, base_sync_done=base_sync_done, use_shm=use_shm
            )
        if not base_sync_done:
            raise ValueError('DS4.1 adapters require the frozen release base already loaded')
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
            BucketedWeightReceiver,
        )

        device = self.device if self.device is not None else self.model_runner.device
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(), device=device, use_shm=use_shm
        )
        return receive_adapter(receiver, self.model_runner, peft_config)

    update._mlite_ds41_adapter = True
    cls.update_weights_from_ipc = update
    return True
