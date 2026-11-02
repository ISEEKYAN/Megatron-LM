# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Byte image of the vLLM ``weights`` pool on node-local storage, for adapter-mode level-2 sleep.

With LoRA served as an adapter the rollout base never changes, so a level-2 wake does not need
the checkpoint loader: it can copy the same pool bytes back, minus the ranges vLLM restores itself
(preserved parameters such as the drafter, and buffers). The first sleep writes one image per
rank with O_DIRECT (no page cache, nothing kept in host memory); every later wake maps the pool
and reads the image back through a few pinned bounce buffers. The adapter publish that follows
the wake rewrites the LoRA slots, as it does after a native reload.
"""
import ctypes
import gc
import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from fnmatch import filter as matching

import torch

from ..c8_fingerprint import checksum

_ALIGN = 4096


def _pool():
    from vllm.device_allocator.cumem import CuMemAllocator

    pool = sorted(
        (int(d.handle[2]), int(d.handle[1]))
        for d in CuMemAllocator.get_instance().pointer_to_data.values()
        if d.tag == "weights"
    )
    if not pool:
        raise RuntimeError("weights image needs the CuMem weights pool (enable_sleep_mode)")
    for _, size in pool:
        if size % _ALIGN:
            raise RuntimeError(f"weights pool allocation of {size} bytes is not O_DIRECT aligned")
    return pool


class _Raw:
    def __init__(self, ptr, size):
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def _sums(pool):
    torch.cuda.synchronize()
    return [checksum(torch.as_tensor(_Raw(ptr, size), device="cuda")) for ptr, size in pool]


def _native_ranges(worker):
    """Device byte ranges that vLLM's own level-2 wake restores (preserved params, buffers)."""
    model = worker.model_runner.model
    draft = worker.get_draft_model()
    parameters = worker._get_sleep_parameters(model)
    names = set()
    for pattern in worker.model_config.sleep_preserve_parameter_names or ():
        names.update(matching(parameters, pattern))
    tensors = [parameters[name] for name in names] + list(model.buffers())
    if draft is not None:
        tensors += list(draft.buffers())
    ranges = []
    for t in tensors:
        if t.is_cuda and t.untyped_storage().nbytes():
            lo = t.untyped_storage().data_ptr()
            ranges.append((lo, lo + t.untyped_storage().nbytes()))
    return sorted(ranges)


def _segments(pool, skip):
    """[(ptr, size)] of the pool minus page-aligned interiors of ``skip`` ranges."""
    inner = sorted(
        (-(-lo // _ALIGN) * _ALIGN, hi // _ALIGN * _ALIGN)
        for lo, hi in skip
        if hi // _ALIGN * _ALIGN > -(-lo // _ALIGN) * _ALIGN
    )
    out, i = [], 0
    for base, size in pool:
        cur, end = base, base + size
        while i < len(inner) and inner[i][1] <= cur:
            i += 1
        j = i
        while j < len(inner) and inner[j][0] < end:
            lo, hi = inner[j]
            if lo > cur:
                out.append((cur, lo - cur))
            cur = max(cur, hi)
            j += 1
        if cur < end:
            out.append((cur, end - cur))
    return out


def _paths(worker, directory):
    rank = worker.vllm_config.parallel_config.data_parallel_rank
    stem = os.path.join(directory, f"{socket.gethostname()}_dp{rank}")
    return stem + ".img", stem + ".json"


def _bounce(count, chunk):
    if chunk % _ALIGN:
        raise ValueError(f"weights image chunk must be a multiple of {_ALIGN} bytes")
    # cudaHostAlloc memory is page aligned, as O_DIRECT requires.
    return [torch.empty(chunk, dtype=torch.uint8, pin_memory=True) for _ in range(count)]


def _cuda():
    return ctypes.CDLL("libcuda.so.1")


def dump(worker, directory, chunk_bytes):
    """Write this rank's weights pool to ``directory``; returns a receipt."""
    started = time.monotonic()
    # Native sleep empties the allocator cache; snapshot the same set of segments it keeps.
    gc.collect()
    torch.cuda.empty_cache()
    pool = _pool()
    sums = _sums(pool)
    segments = _segments(pool, _native_ranges(worker))
    image, manifest = _paths(worker, directory)
    os.makedirs(directory, exist_ok=True)
    cu, (buf,) = _cuda(), _bounce(1, chunk_bytes)
    view = memoryview(buf.numpy())
    fd = os.open(image, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o600)
    try:
        for ptr, size in segments:
            for off in range(0, size, chunk_bytes):
                n = min(chunk_bytes, size - off)
                rc = cu.cuMemcpyDtoH_v2(
                    ctypes.c_void_p(buf.data_ptr()), ctypes.c_uint64(ptr + off), ctypes.c_size_t(n)
                )
                if rc != 0:
                    raise RuntimeError(f"weights image copy-out failed ({rc}) at {hex(ptr + off)}")
                if os.write(fd, view[:n]) != n:
                    raise OSError(f"short write to {image}")
        os.fsync(fd)
    finally:
        os.close(fd)
    total = sum(size for _, size in segments)
    with open(manifest, "w") as f:
        json.dump({"pool": pool, "segments": segments, "sums": sums}, f)
    return {
        "image": image,
        "bytes": total,
        "native_bytes": sum(size for _, size in pool) - total,
        "seconds": time.monotonic() - started,
    }


def restore(worker, directory, chunk_bytes, readers, verify):
    """Read this rank's image back into the freshly mapped weights pool; returns a receipt."""
    started = time.monotonic()
    image, manifest = _paths(worker, directory)
    with open(manifest) as f:
        expected = json.load(f)
    pool = _pool()
    if [list(p) for p in pool] != expected["pool"]:
        raise RuntimeError("weights pool layout changed since the image was written")
    jobs, offset = [], 0
    for ptr, size in expected["segments"]:
        for off in range(0, size, chunk_bytes):
            n = min(chunk_bytes, size - off)
            jobs.append((offset, ptr + off, n))
            offset += n
    cu, buffers = _cuda(), _bounce(2 * readers, chunk_bytes)
    views = [memoryview(b.numpy()) for b in buffers]
    fd = os.open(image, os.O_RDONLY | os.O_DIRECT)

    def read(slot, file_offset, n):
        if os.preadv(fd, [views[slot][:n]], file_offset) != n:
            raise OSError(f"short read from {image} at {file_offset}")
        return slot

    try:
        with ThreadPoolExecutor(readers) as pool_exec:
            pending = [
                pool_exec.submit(read, i, *jobs[i][::2])
                for i in range(min(len(buffers), len(jobs)))
            ]
            for i, (_, dst, n) in enumerate(jobs):
                slot = pending[i].result()
                rc = cu.cuMemcpyHtoD_v2(
                    ctypes.c_uint64(dst),
                    ctypes.c_void_p(buffers[slot].data_ptr()),
                    ctypes.c_size_t(n),
                )
                if rc != 0:
                    raise RuntimeError(f"weights image copy-in failed ({rc}) at {hex(dst)}")
                follow = i + len(buffers)
                if follow < len(jobs):
                    pending.append(pool_exec.submit(read, slot, *jobs[follow][::2]))
    finally:
        os.close(fd)
    copied = time.monotonic() - started
    if verify:
        changed = [i for i, s in enumerate(_sums(pool)) if s != expected["sums"][i]]
        if changed:
            raise RuntimeError(f"weights image restore differs in allocations {changed}")
    return {
        "bytes": offset,
        "copy_seconds": copied,
        "seconds": time.monotonic() - started,
        "verified": bool(verify),
    }


def install():
    """Expose dump/restore as collective_rpc methods of VERL's colocate worker extension."""
    try:
        from verl.workers.rollout.vllm_rollout import utils
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith(("verl", "vllm")):
            return False
        raise
    cls = getattr(utils, "vLLMColocateWorkerExtension", None)
    if cls is None:
        return False

    def mlite_weights_image_dump(self, directory, chunk_bytes):
        return dump(self, directory, chunk_bytes)

    def mlite_weights_image_restore(self, directory, chunk_bytes, readers, verify):
        return restore(self, directory, chunk_bytes, readers, verify)

    cls.mlite_weights_image_dump = mlite_weights_image_dump
    cls.mlite_weights_image_restore = mlite_weights_image_restore
    return True
