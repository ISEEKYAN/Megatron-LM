# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Park the actor before native hybrid rollout wake; reuse engine storage moves."""
import gc
import json
import os
import sys
from functools import wraps

import torch


def record(engine, phase):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    report = dict(
        phase=phase,
        pid=os.getpid(),
        allocated_bytes=allocated,
        reserved_bytes=torch.cuda.memory_reserved(),
        free_bytes=torch.cuda.mem_get_info()[0],
    )
    import pynvml

    pynvml.nvmlInit()
    process_bytes = 0
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if process.pid == os.getpid():
                process_bytes += process.usedGpuMemory
    report['process_GPU_bytes'] = process_bytes
    print('MLITE_EXCLUSIVE_ACTOR ' + json.dumps(report), flush=True)
    if phase == 'before_rollout_wake':
        assert process_bytes <= 8 * 2**30, ('ACTOR_PROCESS_RESIDENCY_ABOVE_8GIB', report)
        assert allocated <= 4 * 2**30, ('ACTOR_GPU_RESIDENCY_ABOVE_4GIB', report)

        # Core DDP keeps CUDA parameter views with zero-sized storage while
        # their actual bytes live in pinned CPU buffer copies.
        def resident(value):
            if value.device.type != 'cuda':
                return 0
            if hasattr(value, 'codes') and hasattr(value, 'scale'):
                return resident(value.codes) + resident(value.scale)
            return value.untyped_storage().nbytes()

        assert not any(resident(p) for p in engine.module.parameters())
        assert not any(resident(b) for b in engine.module.buffers())


def extra_storage(engine, device):
    """Move non-DDP buffers/attributes too; frozen native codes stay native."""
    if device == 'cpu':
        paths = list(getattr(engine, '_exclusive_extra_storage', ()))
        for module in engine.module.modules():
            for name, value in list(module._buffers.items()):
                if value is not None and value.is_cuda:
                    paths.append((module, name, True))
                    module._buffers[name] = value.to('cpu')
            for name, value in list(vars(module).items()):
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    paths.append((module, name, False))
                    setattr(module, name, value.to('cpu'))
        engine._exclusive_extra_storage = paths
    else:
        for module, name, buffer in getattr(engine, '_exclusive_extra_storage', ()):
            if buffer:
                module._buffers[name] = module._buffers[name].to(device)
            else:
                setattr(module, name, getattr(module, name).to(device))
        engine._exclusive_extra_storage = []


def clear_deployment_scratch(engine):
    prefix = 'megatron.lite.model.deepseek_v41.vllm.'
    lora = sys.modules.get(prefix + 'lora')
    if lora is not None:
        lora.invalidate_scaled_b(engine.module, release=True)
    weights = sys.modules.get(prefix + 'weights')
    if weights is not None:
        weights.clear(enabled=False)
    for name, attr in [('batching', '_PACKED'), ('experts', '_BUFFERS')]:
        module = sys.modules.get(prefix + name)
        if module is not None:
            getattr(module, attr).clear()


def consume_host_snapshot(engine):
    for name in ('_exclusive_adapter_snapshot', '_exclusive_initial_adapter'):
        cached = getattr(engine, name, None)
        if cached is not None:
            setattr(engine, name, None)
            return iter(cached[0]), cached[1]
    return None


def install():
    # Engine construction happens after this worker class has been defined.
    from verl.workers import engine_workers

    cls = engine_workers.ActorRolloutRefWorker
    original = cls.update_weights
    if getattr(original, '_mlite_exclusive_actor', False):
        return

    @wraps(original)
    async def update_weights(self, global_steps=None, mode='auto'):
        engine = self.actor.engine
        if not getattr(engine.engine_config, 'exclusive_rollout', False):
            return await original(self, global_steps=global_steps, mode=mode)
        effective = mode if mode != 'auto' else self.config.rollout.checkpoint_engine.backend
        if effective != 'naive' or not self.base_sync_done:
            raise ValueError('Exclusive actor requires native colocated adapter-only sync')
        if engine.is_param_offload_enabled:
            raise ValueError('Exclusive actor uses explicit boundaries, not auto param_offload')
        if getattr(engine, '_exclusive_adapter_snapshot', None) is not None:
            raise RuntimeError('Unconsumed exclusive adapter snapshot')
        weights, peft = engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=True
        )
        if peft is None:
            raise ValueError('Exclusive actor supports adapter-only publication')
        owned = list(weights)
        if not owned or any(value.device.type != 'cpu' for _, value in owned):
            raise ValueError('Adapter snapshot must own host storage before actor offload')
        engine._exclusive_adapter_snapshot = (owned, peft)
        try:
            record(engine, 'actor_before_offload')
            engine.to('cpu', model=True, optimizer=True, grad=True)
            engine._exclusive_actor_offloaded = True
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            record(engine, 'before_rollout_wake')
            # Original native RPC order/IPC/ACK remains authoritative. Its
            # get_per_tensor_param consumes the host snapshot without onload.
            return await original(self, global_steps=global_steps, mode=mode)
        finally:
            engine._exclusive_adapter_snapshot = None

    update_weights._mlite_exclusive_actor = True
    cls.update_weights = update_weights
