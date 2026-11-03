# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Park the actor before native hybrid rollout wake; reuse engine storage moves."""
import gc
import json
import os
import sys
import time
from functools import wraps

import torch


def guard(ok, name, report):
    """Memory guards warn by default; ``MLITE_EXCLUSIVE_GUARD=raise`` makes them fatal."""
    if ok:
        return
    mode = os.environ.get('MLITE_EXCLUSIVE_GUARD', 'warn')
    if mode == 'raise':
        raise AssertionError(name, report)
    if mode != 'warn':
        raise ValueError(f'MLITE_EXCLUSIVE_GUARD must be warn or raise, got {mode!r}')
    print('MLITE_EXCLUSIVE_GUARD_WARN ' + json.dumps(dict(guard=name, **report)), flush=True)


def record(engine, phase):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    report = dict(
        phase=phase,
        pid=os.getpid(),
        allocated_bytes=allocated,
        max_allocated_bytes=torch.cuda.max_memory_allocated(),
        max_reserved_bytes=torch.cuda.max_memory_reserved(),
        reserved_bytes=torch.cuda.memory_reserved(),
        free_bytes=torch.cuda.mem_get_info()[0],
        total_bytes=torch.cuda.mem_get_info()[1],
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
    # NVML includes native communication/runtime allocations outside Torch's
    # allocator. Do not claim an unsupported split between individual libraries.
    report['non_torch_unclassified_bytes'] = max(0, process_bytes - report['reserved_bytes'])
    report['torch_inactive_reserved_bytes'] = report['reserved_bytes'] - allocated
    # Full-model rollout measured ~117 GiB; conservatively reserve 128 GiB.
    # Keep 16 GiB genuinely free even with actor's native process buffers.
    rollout_budget = 128 * 2**30
    margin = 16 * 2**30
    report['rollout_peak_budget_bytes'] = rollout_budget
    report['required_free_margin_bytes'] = margin
    print('MLITE_EXCLUSIVE_ACTOR ' + json.dumps(report), flush=True)
    if phase == 'before_rollout_wake':
        guard(process_bytes <= 16 * 2**30, 'ACTOR_PROCESS_RESIDENCY_ABOVE_16GIB', report)
        guard(allocated <= 4 * 2**30, 'ACTOR_GPU_RESIDENCY_ABOVE_4GIB', report)
        guard(
            process_bytes + rollout_budget + margin <= report['total_bytes'],
            'ACTOR_ROLLOUT_CAPACITY_BUDGET',
            report,
        )
        guard(report['free_bytes'] >= rollout_budget + margin, 'ROLLOUT_WAKE_FREE_MARGIN', report)

        # Core DDP keeps CUDA parameter views with zero-sized storage while
        # their actual bytes live in pinned CPU buffer copies.
        def resident(value):
            if value.device.type != 'cuda':
                return 0
            if hasattr(value, 'codes') and hasattr(value, 'scale'):
                return resident(value.codes) + resident(value.scale)
            return value.untyped_storage().nbytes()

        guard(
            not any(resident(p) for p in engine.module.parameters()),
            'ACTOR_PARAMETER_RESIDENT',
            report,
        )
        guard(
            not any(resident(b) for _, b in engine.module.named_buffers()),
            'ACTOR_BUFFER_RESIDENT',
            report,
        )


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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
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
        start, start_mono = time.time(), time.perf_counter()
        weights, peft = engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=True
        )
        if peft is None:
            raise ValueError('Exclusive actor supports adapter-only publication')
        owned = list(weights)
        if not owned or any(value.device.type != 'cpu' for _, value in owned):
            raise ValueError('Adapter snapshot must own host storage before actor offload')
        engine._exclusive_adapter_snapshot = (owned, peft)
        times = [('snapshot', time.perf_counter())]
        try:
            record(engine, 'actor_before_offload')
            times.append(('record', time.perf_counter()))
            engine.to('cpu', model=True, optimizer=True, grad=True)
            engine._exclusive_actor_offloaded = True
            times.append(('offload', time.perf_counter()))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            times.append(('release', time.perf_counter()))
            record(engine, 'before_rollout_wake')
            # One server wake restores all DEP workers. Every actor rank
            # must be parked before the first caller wakes that shared server.
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            times.append(('barrier', time.perf_counter()))
            # Original native RPC order/IPC/ACK remains authoritative. Its
            # get_per_tensor_param consumes the host snapshot without onload.
            result = await original(self, global_steps=global_steps, mode=mode)
            times.append(('wake_publish', time.perf_counter()))
            return result
        finally:
            engine._exclusive_adapter_snapshot = None
            receipt = dict(pid=os.getpid(), global_steps=global_steps, start=start)
            receipt['snapshot'] = round(times[0][1] - start_mono, 3)
            receipt.update(
                (name, round(t - prev, 3)) for (_, prev), (name, t) in zip(times, times[1:])
            )
            receipt['seconds'] = round(time.perf_counter() - start_mono, 3)
            print('MLITE_EXCLUSIVE_SYNC ' + json.dumps(receipt), flush=True)

    update_weights._mlite_exclusive_actor = True
    cls.update_weights = update_weights
