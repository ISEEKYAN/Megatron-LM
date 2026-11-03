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


def _split(segments, host_bytes):
    """Move the first ``host_bytes`` (page aligned) of ``segments`` into a host-resident part."""
    host_bytes = host_bytes // _ALIGN * _ALIGN
    host, disk = [], []
    for ptr, size in segments:
        take = min(size, host_bytes)
        if take:
            host.append((ptr, take))
            host_bytes -= take
        if take < size:
            disk.append((ptr + take, size - take))
    return host, disk


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


def dump(worker, directory, chunk_bytes, host_bytes):
    """Write this rank's weights pool to ``directory``, keeping ``host_bytes`` in pinned host
    memory (restored in parallel with the disk read); returns a receipt."""
    started = time.monotonic()
    # Native sleep empties the allocator cache; snapshot the same set of segments it keeps.
    gc.collect()
    torch.cuda.empty_cache()
    pool = _pool()
    sums = _sums(pool)
    host_segments, segments = _split(_segments(pool, _native_ranges(worker)), host_bytes)
    image, manifest = _paths(worker, directory)
    os.makedirs(directory, exist_ok=True)
    cu, (buf,) = _cuda(), _bounce(1, chunk_bytes)
    kept = []
    for ptr, size in host_segments:
        # Exact-size page-locked copy (pin_memory() rounds each block up to a power of 2).
        host = torch.empty(size, dtype=torch.uint8)
        torch.cuda.check_error(torch.cuda.cudart().cudaHostRegister(host.data_ptr(), size, 0))
        rc = cu.cuMemcpyDtoH_v2(
            ctypes.c_void_p(host.data_ptr()), ctypes.c_uint64(ptr), ctypes.c_size_t(size)
        )
        if rc != 0:
            raise RuntimeError(f"weights image host copy-out failed ({rc}) at {hex(ptr)}")
        kept.append((ptr, host))
    worker._mlite_weights_image_host = kept
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
    host_total = sum(size for _, size in host_segments)
    with open(manifest, "w") as f:
        json.dump(
            {"pool": pool, "host_segments": host_segments, "segments": segments, "sums": sums}, f
        )
    return {
        "image": image,
        "bytes": total,
        "host_bytes": host_total,
        "native_bytes": sum(size for _, size in pool) - total - host_total,
        "seconds": time.monotonic() - started,
    }


def _tensors(owner):
    """CUDA tensors held directly by ``owner``'s attributes (or in a list, tuple or dict)."""
    for value in vars(owner).values():
        if isinstance(value, dict):
            values = value.values()
        elif isinstance(value, (list, tuple)):
            values = value
        else:
            values = (value,)
        for t in values:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                yield t


def _lora_ranges(worker):
    """Device byte ranges an adapter publish may write: every tensor owned by a LoRA layer
    (slots, per-expert slot copies) or its punica wrapper, kernel views of the adapter
    (``_views``), and any attribute named like a LoRA tensor."""
    tensors = []
    for module in worker.model_runner.model.modules():
        if callable(getattr(module, "set_lora", None)):
            tensors += _tensors(module)
            tensors += list(module.parameters(recurse=False)) + list(module.buffers(recurse=False))
            punica = getattr(module, "punica_wrapper", None)
            if punica is not None:
                tensors += _tensors(punica)
        views = getattr(module, "_views", None)
        if isinstance(views, dict):
            tensors += [t for t in views.values() if isinstance(t, torch.Tensor) and t.is_cuda]
        for name, value in vars(module).items():
            if "lora" in name.lower():
                values = value if isinstance(value, (list, tuple)) else (value,)
                tensors += [t for t in values if isinstance(t, torch.Tensor) and t.is_cuda]
    ranges = set()
    for t in tensors:
        if t.untyped_storage().nbytes():
            lo = t.untyped_storage().data_ptr()
            ranges.add((lo, lo + t.untyped_storage().nbytes()))
    return sorted(ranges)


def _copy_in(cu, fd, image, jobs, buffers, readers):
    """Read ``jobs`` [(file_offset, device_ptr, n)] from ``fd`` through the bounce buffers."""
    views = [memoryview(b.numpy()) for b in buffers]

    def read(slot, file_offset, n):
        if os.preadv(fd, [views[slot][:n]], file_offset) != n:
            raise OSError(f"short read from {image} at {file_offset}")
        return slot

    with ThreadPoolExecutor(readers) as pool_exec:
        pending = [
            pool_exec.submit(read, i, *jobs[i][::2]) for i in range(min(len(buffers), len(jobs)))
        ]
        for i, (_, dst, n) in enumerate(jobs):
            slot = pending[i].result()
            rc = cu.cuMemcpyHtoD_v2(
                ctypes.c_uint64(dst), ctypes.c_void_p(buffers[slot].data_ptr()), ctypes.c_size_t(n)
            )
            if rc != 0:
                raise RuntimeError(f"weights image copy-in failed ({rc}) at {hex(dst)}")
            follow = i + len(buffers)
            if follow < len(jobs):
                pending.append(pool_exec.submit(read, slot, *jobs[follow][::2]))


def _verify(pool, expected, indices):
    sums = _sums([pool[i] for i in indices])
    changed = [i for i, s in zip(indices, sums) if s != expected["sums"][i]]
    if changed:
        raise RuntimeError(f"weights image restore differs in allocations {changed}")


def restore(worker, directory, chunk_bytes, readers, verify, background=False):
    """Read this rank's image back into the freshly mapped weights pool; returns a receipt.

    With ``background`` only the allocations holding LoRA slots are restored (and verified)
    before returning, so the adapter publish can rewrite those slots while a thread restores
    the rest; ``join`` waits for that thread and verifies the remaining allocations.
    """
    started = time.monotonic()
    image, manifest = _paths(worker, directory)
    with open(manifest) as f:
        expected = json.load(f)
    pool = [tuple(p) for p in _pool()]
    if [list(p) for p in pool] != expected["pool"]:
        raise RuntimeError("weights pool layout changed since the image was written")
    kept = getattr(worker, "_mlite_weights_image_host", [])
    if [[ptr, host.numel()] for ptr, host in kept] != expected.get("host_segments", []):
        raise RuntimeError("weights image host part does not match its manifest")
    jobs, offset = [], 0
    for ptr, size in expected["segments"]:
        for off in range(0, size, chunk_bytes):
            n = min(chunk_bytes, size - off)
            jobs.append((offset, ptr + off, n))
            offset += n
    first = range(len(pool))
    if background:
        lora = _lora_ranges(worker)
        first = [
            i
            for i, (base, size) in enumerate(pool)
            if any(lo < base + size and hi > base for lo, hi in lora)
        ]
    early = [(pool[i][0], pool[i][0] + pool[i][1]) for i in first]

    def owned(dst):
        return any(lo <= dst < hi for lo, hi in early)

    now = [job for job in jobs if owned(job[1])] if background else jobs
    later = [job for job in jobs if not owned(job[1])] if background else []
    cu, buffers = _cuda(), _bounce(2 * readers, chunk_bytes)
    if background:
        # Driver calls on the restore thread need this context made current there too.
        context = ctypes.c_void_p()
        rc = cu.cuCtxGetCurrent(ctypes.byref(context))
        if rc != 0 or not context.value:
            raise RuntimeError(f"weights image restore has no current CUDA context ({rc})")
    fd = os.open(image, os.O_RDONLY | os.O_DIRECT)
    try:
        # The host part is a few GiB over C2C; copy all of it before the disk part.
        for ptr, host in kept:
            rc = cu.cuMemcpyHtoD_v2(
                ctypes.c_uint64(ptr),
                ctypes.c_void_p(host.data_ptr()),
                ctypes.c_size_t(host.numel()),
            )
            if rc != 0:
                raise RuntimeError(f"weights image host copy-in failed ({rc}) at {hex(ptr)}")
        _copy_in(cu, fd, image, now, buffers, readers)
    except BaseException:
        os.close(fd)
        raise
    receipt = {
        "bytes": offset,
        "host_bytes": sum(host.numel() for _, host in kept),
        "verified": bool(verify),
    }
    if not background:
        os.close(fd)
        receipt["copy_seconds"] = time.monotonic() - started
        if verify:
            _verify(pool, expected, range(len(pool)))
        receipt["seconds"] = time.monotonic() - started
        return receipt
    if verify:
        _verify(pool, expected, first)
    receipt["early_bytes"] = sum(n for _, _, n in now)
    receipt["early_allocations"] = len(first)
    receipt["early_seconds"] = time.monotonic() - started
    done = set(first)
    rest = [i for i in range(len(pool)) if i not in done]

    def finish():
        try:
            if cu.cuCtxSetCurrent(context) != 0:
                raise RuntimeError("weights image restore thread cannot enter the CUDA context")
            _copy_in(cu, fd, image, later, buffers, readers)
        finally:
            os.close(fd)
        receipt["copy_seconds"] = time.monotonic() - started
        if verify:
            _verify(pool, expected, rest)
        receipt["seconds"] = time.monotonic() - started
        return receipt

    worker._mlite_weights_image_join = ThreadPoolExecutor(1).submit(finish)
    return dict(receipt)


def join(worker):
    """Wait for a background restore; returns its receipt."""
    future = getattr(worker, "_mlite_weights_image_join", None)
    if future is None:
        raise RuntimeError("no background weights image restore to join")
    worker._mlite_weights_image_join = None
    return future.result()


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

    def mlite_weights_image_dump(self, directory, chunk_bytes, host_bytes):
        return dump(self, directory, chunk_bytes, host_bytes)

    def mlite_weights_image_restore(self, directory, chunk_bytes, readers, verify):
        return restore(self, directory, chunk_bytes, readers, verify)

    def mlite_weights_image_restore_start(self, directory, chunk_bytes, readers, verify):
        return restore(self, directory, chunk_bytes, readers, verify, background=True)

    def mlite_weights_image_restore_join(self):
        return join(self)

    cls.mlite_weights_image_dump = mlite_weights_image_dump
    cls.mlite_weights_image_restore = mlite_weights_image_restore
    cls.mlite_weights_image_restore_start = mlite_weights_image_restore_start
    cls.mlite_weights_image_restore_join = mlite_weights_image_restore_join
    return True
