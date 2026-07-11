# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Load the DeepSeek V4 vLLM rollout model without starting RL workers."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from pathlib import Path


def _checkpoint_ipc_handle(worker):
    return worker._get_zmq_handle()


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


def probe_checkpoint_sync(llm, *, worker_count: int) -> None:
    handles = llm.collective_rpc(_checkpoint_ipc_handle, timeout=300)
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
    if sync_probe:
        probe_checkpoint_sync(llm, worker_count=rollout_tp)
    print(
        "DS4_VLLM_LOAD_ONLY_PASSED "
        f"rollout_tp={rollout_tp} o_groups={o_groups} "
        f"local_groups={o_groups // rollout_tp}",
        flush=True,
    )


if __name__ == "__main__":
    main()
