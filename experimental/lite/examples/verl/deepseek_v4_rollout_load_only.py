# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Load the DeepSeek V4 vLLM rollout model without starting RL workers."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from pathlib import Path


def _send_empty_checkpoint_bucket(handle: str, ready: Event) -> None:
    from multiprocessing.shared_memory import SharedMemory

    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 300_000)
    socket.setsockopt(zmq.SNDTIMEO, 300_000)
    shared = SharedMemory(create=True, size=1)
    try:
        socket.bind(handle)
        ready.set()
        socket.send_pyobj({"name": shared.name, "size": shared.size})
        socket.recv()
        socket.send_pyobj({"bucket_meta": {}, "is_last": True})
        socket.recv()
    finally:
        socket.close()
        context.term()
        shared.close()
        shared.unlink()


def _parse_compute_apps(csv_text: str, uuid_to_index: dict[str, int]) -> dict[int, list[tuple[int, int]]]:
    """Parse ``nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory`` CSV
    (``noheader,nounits``) into ``{gpu_index: [(pid, mib), ...]}``.

    Split from the subprocess call so the residency accounting is unit-testable
    without a GPU. Lines whose GPU uuid is unknown or that are malformed are
    skipped rather than raising, so a partial ``nvidia-smi`` snapshot still yields
    the residency it can attribute.
    """
    residency: dict[int, list[tuple[int, int]]] = {}
    for line in csv_text.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        uuid, pid, mem = parts
        index = uuid_to_index.get(uuid)
        if index is None:
            continue
        mem_fields = mem.split()
        mib = int(mem_fields[0]) if mem_fields and mem_fields[0].isdigit() else 0
        residency.setdefault(index, []).append((int(pid), mib))
    return residency


def dump_gpu_residency(tag: str) -> None:
    """Emit per-GPU per-process residency so the 8-GPU load-only A/B/C experiment
    can read whether the sibling vLLM TP ranks leak a CUDA context onto peer GPUs
    (root cause of the 128-GPU resync GPU0 OOM; see
    docs/ds4-resync-gpu0-sibling-residency.md). Opt-in via
    ``MLITE_VLLM_RESIDENCY_PROBE=1`` — never runs on the hot path.
    """
    import subprocess

    uuid_csv = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
    )
    uuid_to_index: dict[str, int] = {}
    for line in uuid_csv.strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit():
            uuid_to_index[fields[1]] = int(fields[0])
    apps_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    residency = _parse_compute_apps(apps_csv, uuid_to_index)
    for index in sorted(residency):
        procs = residency[index]
        total_mib = sum(mib for _, mib in procs)
        detail = " ".join(f"{pid}:{mib}MiB" for pid, mib in procs)
        print(
            f"MLITE_RESIDENCY_PROBE tag={tag} gpu={index} nproc={len(procs)} "
            f"total_mib={total_mib} {detail}",
            flush=True,
        )


def probe_checkpoint_sync(llm, *, worker_count: int) -> None:
    handles = llm.collective_rpc("_get_zmq_handle", timeout=300)
    if len(handles) != worker_count or len(set(handles)) != worker_count:
        raise RuntimeError(
            f"expected {worker_count} unique checkpoint IPC handles, got {handles}"
        )

    ready = [Event() for _ in handles]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_send_empty_checkpoint_bucket, handle, event)
            for handle, event in zip(handles, ready, strict=True)
        ]
        for event, future in zip(ready, futures, strict=True):
            if not event.wait(timeout=30):
                future.result()
                raise TimeoutError("checkpoint IPC sender did not bind within 30 seconds")
        results = llm.collective_rpc(
            "update_weights_from_ipc",
            timeout=300,
            kwargs={
                "peft_config": None,
                "base_sync_done": True,
                "use_shm": True,
            },
        )
        for future in futures:
            future.result()

    if results != [None] * worker_count:
        raise RuntimeError(f"unexpected checkpoint sync results: {results}")
    print(
        "DS4_VLLM_CHECKPOINT_SYNC_PASSED "
        f"workers={worker_count} base_sync_done=true buckets=1 tensors=0",
        flush=True,
    )


def main() -> None:
    checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"])
    rollout_tp = int(os.environ["ROLLOUT_TP"])
    with (checkpoint_dir / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    o_groups = config.get("o_groups")
    if not isinstance(o_groups, int) or o_groups < 1:
        raise RuntimeError(f"DeepSeek V4 config has invalid o_groups={o_groups!r}")
    if rollout_tp < 1 or o_groups % rollout_tp != 0:
        raise RuntimeError(
            f"DeepSeek V4 o_groups={o_groups} must be divisible by "
            f"positive rollout_tp={rollout_tp}"
        )

    from vllm import LLM

    sync_probe = os.environ.get("VLLM_CHECKPOINT_SYNC_PROBE", "0") == "1"
    llm_kwargs = {}
    if sync_probe:
        llm_kwargs["worker_extension_cls"] = (
            "verl_mlite.rollout.verl_worker.VllmCheckpointWorkerExtension"
        )

    llm = LLM(
        model=str(checkpoint_dir),
        tensor_parallel_size=rollout_tp,
        load_format="dummy",
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=384,
        max_num_seqs=32,
        max_num_batched_tokens=4096,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=float(
            os.environ.get("ROLLOUT_GPU_MEMORY_UTILIZATION", "0.60")
        ),
        kv_cache_dtype="fp8",
        enforce_eager=True,
        disable_log_stats=True,
        hf_overrides={
            "expert_dtype": "fp8",
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "scale_fmt": "float32",
                "weight_block_size": [128, 128],
            },
        },
        **llm_kwargs,
    )
    if os.environ.get("MLITE_VLLM_RESIDENCY_PROBE") == "1":
        dump_gpu_residency("post_init")
    if sync_probe:
        probe_checkpoint_sync(llm, worker_count=rollout_tp)
        if os.environ.get("MLITE_VLLM_RESIDENCY_PROBE") == "1":
            dump_gpu_residency("post_sync")
    print(
        "DS4_VLLM_LOAD_ONLY_PASSED "
        f"rollout_tp={rollout_tp} o_groups={o_groups} "
        f"local_groups={o_groups // rollout_tp}",
        flush=True,
    )


if __name__ == "__main__":
    main()
