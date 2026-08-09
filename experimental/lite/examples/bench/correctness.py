# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# isort: skip_file
"""Deterministic correctness runner for MLite and reference backends."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import struct
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

_EXPERIMENTAL_LITE_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_EXPERIMENTAL_LITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTAL_LITE_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(_REPO_ROOT))

from examples.bench.bench import (
    BenchCliConfig,
    build_runtime_config,
    build_session_config,
)
from examples.bench.results import compare_correctness_artifacts, load_result_artifact
from examples.bench.session import _make_data_iter
from megatron.lite.primitive.deterministic import set_deterministic
from megatron.lite.runtime import create_runtime


def _distributed_rank() -> int:
    for name in ("RANK", "SLURM_PROCID"):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return 0


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _scalar(value: float | int | torch.Tensor | None) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("scalar fingerprint requires a scalar tensor.")
        value = float(value.detach().cpu().float().item())
    value_f = float(0.0 if value is None else value)
    return {
        "value": value_f,
        "float_hex": value_f.hex(),
        "sha256_f64_be": hashlib.sha256(struct.pack(">d", value_f)).hexdigest(),
    }


def _hash_tensor(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    t = tensor.detach().contiguous().cpu()
    raw = t.view(torch.uint8).numpy().tobytes()
    as_bf16 = t.to(torch.bfloat16).contiguous() if t.is_floating_point() else None
    summary = t.float() if t.is_floating_point() else None
    result = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if as_bf16 is not None:
        result["sha256_as_bf16"] = hashlib.sha256(
            as_bf16.view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        if os.environ.get("MLITE_CORRECTNESS_INCLUDE_VALUES") == "1":
            result["values"] = [float(x) for x in t.float().reshape(-1).tolist()]
    if summary is not None:
        flat = summary.reshape(-1)
        result["summary"] = {
            "min": float(flat.min().item()) if flat.numel() else 0.0,
            "max": float(flat.max().item()) if flat.numel() else 0.0,
            "mean": float(flat.mean().item()) if flat.numel() else 0.0,
            "first8": [float(x) for x in flat[:8].tolist()],
        }
    return result


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _record_activation_probe(
    records: list[dict[str, Any]],
    name: str,
    output: Any,
    *,
    record_grad: bool = False,
    tensor_hooks: list[Any] | None = None,
) -> None:
    tensor = _first_tensor(output)
    record = {"name": name, "found": True, "tensor": _hash_tensor(tensor)}
    if record_grad:
        record["grad"] = None
        record["grad_found"] = isinstance(tensor, torch.Tensor) and tensor.requires_grad
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:

            def _grad_hook(grad, _record=record):
                _record["grad"] = _hash_tensor(grad)

            hook = tensor.register_hook(_grad_hook)
            if tensor_hooks is not None:
                tensor_hooks.append(hook)
    records.append(record)


def _resolve_probe_module(modules: dict[str, Any], name: str) -> tuple[str, Any] | None:
    module = modules.get(name)
    if module is not None:
        return name, module
    matches = [
        (candidate_name, candidate)
        for candidate_name, candidate in modules.items()
        if candidate_name.endswith(f".{name}")
    ]
    if len(matches) == 1:
        return matches[0]
    return None


@contextmanager
def _activation_probe_context(
    handle, probe_names: list[str], *, record_grad: bool = False
):
    records: list[dict[str, Any]] = []
    hooks = []
    tensor_hooks = []
    patched_methods = []
    modules = dict(handle._model.named_modules())
    for name in probe_names:
        probe_name = name
        record_input = probe_name.endswith(":input")
        lookup_name = probe_name[:-6] if record_input else probe_name
        if "::" in lookup_name:
            module_name, method_name = lookup_name.split("::", 1)
            resolved = _resolve_probe_module(modules, module_name)
            if resolved is None or not hasattr(resolved[1], method_name):
                records.append({"name": name, "found": False})
                continue
            resolved_name, module = resolved
            original = getattr(module, method_name)

            def _wrapped(
                *args,
                _original=original,
                _probe_name=name,
                _resolved_name=resolved_name,
                _module_type=type(module).__module__ + "." + type(module).__qualname__,
                _record_input=record_input,
                _method_name=method_name,
                **kwargs,
            ):
                if _record_input:
                    _record_activation_probe(
                        records,
                        _probe_name,
                        args,
                        record_grad=record_grad,
                        tensor_hooks=tensor_hooks,
                    )
                    records[-1][
                        "resolved_name"
                    ] = f"{_resolved_name}::{_method_name}:input"
                    records[-1]["module_type"] = _module_type
                    return _original(*args, **kwargs)
                output = _original(*args, **kwargs)
                _record_activation_probe(
                    records,
                    _probe_name,
                    output,
                    record_grad=record_grad,
                    tensor_hooks=tensor_hooks,
                )
                records[-1]["resolved_name"] = f"{_resolved_name}::{_method_name}"
                records[-1]["module_type"] = _module_type
                return output

            setattr(module, method_name, _wrapped)
            patched_methods.append((module, method_name, original))
            continue

        resolved = _resolve_probe_module(modules, lookup_name)
        if resolved is None:
            records.append({"name": name, "found": False})
            continue
        resolved_name, module = resolved

        if record_input:

            def _pre_hook(
                _module, args, probe_name=name, resolved_probe_name=resolved_name
            ):
                _record_activation_probe(
                    records,
                    probe_name,
                    args,
                    record_grad=record_grad,
                    tensor_hooks=tensor_hooks,
                )
                records[-1]["resolved_name"] = f"{resolved_probe_name}:input"
                records[-1]["module_type"] = (
                    type(_module).__module__ + "." + type(_module).__qualname__
                )

            hooks.append(module.register_forward_pre_hook(_pre_hook))
            continue

        def _hook(
            _module, _args, output, probe_name=name, resolved_probe_name=resolved_name
        ):
            _record_activation_probe(
                records,
                probe_name,
                output,
                record_grad=record_grad,
                tensor_hooks=tensor_hooks,
            )
            records[-1]["resolved_name"] = resolved_probe_name
            records[-1]["module_type"] = (
                type(_module).__module__ + "." + type(_module).__qualname__
            )

        hooks.append(module.register_forward_hook(_hook))
    try:
        yield records
    finally:
        for hook in tensor_hooks:
            hook.remove()
        for hook in hooks:
            hook.remove()
        for module, method_name, original in patched_methods:
            setattr(module, method_name, original)


def _update_hash_with_tensor(h: Any, name: str, tensor: torch.Tensor) -> None:
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    t = tensor.detach().contiguous().cpu()
    h.update(name.encode("utf-8"))
    h.update(b"\0")
    h.update(str(t.dtype).encode("ascii"))
    h.update(b"\0")
    h.update(json.dumps(list(t.shape), separators=(",", ":")).encode("ascii"))
    h.update(b"\0")
    h.update(t.view(torch.uint8).numpy().tobytes())
    h.update(b"\0")


def _model_chunks(handle) -> list[Any]:
    chunks = handle._extras.get("model_chunks")
    if chunks is None:
        chunks = handle._extras.get("model_list")
    if chunks is None:
        chunks = [handle._model]
    return list(chunks)


def _grad_fingerprint(handle) -> dict[str, Any]:
    h = hashlib.sha256()
    count = 0
    details = []
    include_details = os.environ.get("MLITE_CORRECTNESS_GRAD_DETAILS") == "1"
    for chunk_idx, chunk in enumerate(_model_chunks(handle)):
        param_sync = getattr(chunk, "param_sync", None)
        if param_sync is None:
            named_params = list(chunk.named_parameters())
        else:
            # M-FSDP's public named_parameters() materializes every bucket
            # after optimizer construction.  Fingerprint the authoritative
            # optimizer shards directly so diagnostics remain bounded.
            named_params = [
                (spec.name, spec.shard_param)
                for bucket in param_sync.buckets
                for spec in bucket.specs
                if spec.shard_param is not None
            ]
        for name, param in sorted(named_params, key=lambda item: item[0]):
            grad = param.grad
            if grad is None:
                grad = getattr(param, "main_grad", None)
            if grad is None:
                continue
            fingerprint_name = f"{chunk_idx}:{name}"
            _update_hash_with_tensor(h, fingerprint_name, grad)
            if include_details:
                detail = _hash_tensor(grad)
                assert detail is not None
                detail["name"] = fingerprint_name
                details.append(detail)
            count += 1
    result = {"sha256": h.hexdigest(), "tensor_count": count}
    if include_details:
        result["details"] = details
    return result


def _weight_fingerprint(rt, handle) -> dict[str, Any]:
    h = hashlib.sha256()
    count = 0
    details = []
    include_details = os.environ.get("MLITE_CORRECTNESS_WEIGHT_DETAILS") == "1"
    for name, tensor in sorted(rt.export_weights(handle), key=lambda item: item[0]):
        _update_hash_with_tensor(h, str(name), tensor)
        if include_details:
            detail = _hash_tensor(tensor)
            assert detail is not None
            detail["name"] = str(name)
            details.append(detail)
        count += 1
    result = {"sha256": h.hexdigest(), "tensor_count": count}
    if include_details:
        result["details"] = details
    return result


def _forward_logits(rt, handle, batch: Any) -> torch.Tensor | None:
    result = rt.forward_backward(
        handle, iter([batch]), loss_fn=None, num_microbatches=1, forward_only=True
    )
    output = result.model_output
    return (
        output.vocab_parallel_logits
        if output.vocab_parallel_logits is not None
        else (-output.log_probs if output.log_probs is not None else None)
    )


def _distributed_max(value: float, device: str) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor.cpu())


def _measure_training_performance(
    rt, handle, data_iter, session_cfg, *, warmup_steps: int, measured_steps: int
) -> dict[str, Any]:
    device_type = torch.device(session_cfg.device).type

    def train_step() -> None:
        rt.zero_grad(handle)
        result = rt.forward_backward(
            handle,
            data_iter,
            loss_fn=None,
            num_microbatches=session_cfg.num_microbatches,
        )
        if not session_cfg.no_optimizer:
            success, _grad_norm, _num_zeros = rt.optimizer_step(handle)
            if not success:
                raise RuntimeError("performance step optimizer update failed")
            rt.lr_scheduler_step(handle)
        if result.model_output.loss is None and result.metrics.get("loss") is None:
            raise RuntimeError("performance step did not produce a loss")

    for _ in range(warmup_steps):
        train_step()
    _sync(session_cfg.device)
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    step_seconds = []
    for _ in range(measured_steps):
        _sync(session_cfg.device)
        started = time.perf_counter()
        train_step()
        _sync(session_cfg.device)
        step_seconds.append(
            _distributed_max(time.perf_counter() - started, session_cfg.device)
        )

    ps = handle._parallel_state
    dp_size = int(getattr(ps, "dp_size", getattr(ps, "dp_cp_size", 1)))
    tokens_per_step = (
        int(session_cfg.seq_len) * int(session_cfg.num_microbatches) * dp_size
    )
    mean_seconds = sum(step_seconds) / len(step_seconds)
    memory = None
    if device_type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        memory = {
            "peak_allocated_bytes_max": int(
                _distributed_max(
                    float(torch.cuda.max_memory_allocated()), session_cfg.device
                )
            ),
            "peak_reserved_bytes_max": int(
                _distributed_max(
                    float(torch.cuda.max_memory_reserved()), session_cfg.device
                )
            ),
            "total_bytes_max": int(
                _distributed_max(float(total_bytes), session_cfg.device)
            ),
            "used_bytes_max": int(
                _distributed_max(float(total_bytes - free_bytes), session_cfg.device)
            ),
        }
    return {
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "step_seconds_max_rank": step_seconds,
        "mean_step_seconds": mean_seconds,
        "tokens_per_step": tokens_per_step,
        "tokens_per_second": tokens_per_step / mean_seconds,
        "memory": memory,
    }


def run_backend(
    cfg: BenchCliConfig,
    *,
    hash_weights: bool = True,
    activation_probe_names: list[str] | None = None,
    verify_resume_path: str | None = None,
    performance_warmup_steps: int = 0,
    performance_steps: int = 0,
) -> dict[str, Any]:
    os.environ["MEGATRON_LITE_DETERMINISTIC"] = "1"
    set_deterministic(cfg.seed)

    rt_cfg = build_runtime_config(cfg)
    rt = create_runtime(rt_cfg)
    handle = rt.build_model()
    session_cfg = build_session_config(cfg)

    activation_probe_names = list(activation_probe_names or [])
    activation_probes: list[dict[str, Any]] = []
    eval_logits = None
    if not cfg.skip_eval:
        eval_iter = _make_data_iter(handle, session_cfg)
        eval_batch = next(eval_iter)
        with _activation_probe_context(
            handle, activation_probe_names
        ) as activation_probes:
            with rt.eval_mode(handle):
                eval_logits = _hash_tensor(_forward_logits(rt, handle, eval_batch))

    data_iter = _make_data_iter(handle, session_cfg)
    steps: list[dict[str, Any]] = []
    with rt.train_mode(handle):
        for step in range(session_cfg.steps):
            with _activation_probe_context(
                handle, activation_probe_names, record_grad=True
            ) as train_activation_probes:
                rt.zero_grad(handle)
                _sync(session_cfg.device)
                result = rt.forward_backward(
                    handle,
                    data_iter,
                    loss_fn=None,
                    num_microbatches=session_cfg.num_microbatches,
                )
                _sync(session_cfg.device)
                output = result.model_output
                logits = _hash_tensor(
                    output.vocab_parallel_logits
                    if output.vocab_parallel_logits is not None
                    else (-output.log_probs if output.log_probs is not None else None)
                )
                grads = _grad_fingerprint(handle)

                if session_cfg.no_optimizer:
                    update_successful, grad_norm, num_zeros = True, 0.0, 0
                else:
                    update_successful, grad_norm, num_zeros = rt.optimizer_step(handle)
                    rt.lr_scheduler_step(handle)
                _sync(session_cfg.device)
                loss = result.metrics.get("loss")
                if loss is None:
                    loss = result.model_output.loss

            steps.append(
                {
                    "step": step,
                    "loss": _scalar(0.0 if loss is None else loss),
                    "logits": logits,
                    "grad_fingerprint": grads,
                    "grad_norm": _scalar(grad_norm),
                    "update_successful": bool(update_successful),
                    "num_zeros": None if num_zeros is None else int(num_zeros),
                    "post_step_weights": (
                        _weight_fingerprint(rt, handle) if hash_weights else None
                    ),
                    "train_activation_probes": train_activation_probes,
                }
            )

    if performance_steps and verify_resume_path is not None:
        raise ValueError(
            "combined performance measurement and resume verification require "
            "separate fresh-process arms"
        )
    performance = (
        _measure_training_performance(
            rt,
            handle,
            data_iter,
            session_cfg,
            warmup_steps=performance_warmup_steps,
            measured_steps=performance_steps,
        )
        if performance_steps
        else None
    )

    resume = None
    if verify_resume_path is not None:
        rt.save_checkpoint(handle, verify_resume_path, global_step=session_cfg.steps)
        next_batches = [next(data_iter) for _ in range(session_cfg.num_microbatches)]

        def next_step(step_rt, step_handle):
            step_rt.zero_grad(step_handle)
            result = step_rt.forward_backward(
                step_handle,
                iter(next_batches),
                loss_fn=None,
                num_microbatches=session_cfg.num_microbatches,
            )
            success, grad_norm, _num_zeros = step_rt.optimizer_step(step_handle)
            step_rt.lr_scheduler_step(step_handle)
            loss = result.metrics.get("loss")
            if loss is None:
                loss = result.model_output.loss
            return {
                "loss": _scalar(loss),
                "grad_norm": _scalar(grad_norm),
                "update_successful": bool(success),
                "post_step_weights": _weight_fingerprint(step_rt, step_handle),
            }

        uninterrupted = next_step(rt, handle)
        rt.to(handle, "cpu", model=True, optimizer=True, grad=True)
        del handle
        gc.collect()
        torch.cuda.empty_cache()

        resumed_rt = create_runtime(rt_cfg)
        resumed_handle = resumed_rt.build_model()
        loaded_step = resumed_rt.load_checkpoint(resumed_handle, verify_resume_path)
        if loaded_step != session_cfg.steps:
            raise AssertionError(
                f"resume step mismatch: {loaded_step} != {session_cfg.steps}"
            )
        resumed = next_step(resumed_rt, resumed_handle)
        resume = {
            "loaded_step": loaded_step,
            "uninterrupted": uninterrupted,
            "resumed": resumed,
            "exact_match": uninterrupted == resumed,
        }
        if not resume["exact_match"]:
            raise AssertionError(
                "checkpoint resume next step differs from uninterrupted run"
            )

    return {
        "kind": "mlite_bench_correctness",
        "backend": cfg.backend,
        "model_name": cfg.model_name,
        "seed": cfg.seed,
        "seq_len": cfg.seq_len,
        "num_microbatches": cfg.num_microbatches,
        "steps": steps,
        "eval_logits": eval_logits,
        "activation_probes": activation_probes,
        "performance": performance,
        "resume": resume,
        "metadata": {
            "deterministic": True,
            "hash_weights": hash_weights,
            "same_data_across_dp": cfg.same_data_across_dp,
            "use_thd": cfg.use_thd,
            "skip_eval": cfg.skip_eval,
        },
    }


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend", choices=["mlite", "bridge", "mbridge"], required=True
    )
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--model-name", default="qwen3_5")
    parser.add_argument("--impl", default="lite")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--etp", type=int, default=None)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--vpp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--performance-warmup-steps", type=int, default=0)
    parser.add_argument("--performance-steps", type=int, default=0)
    parser.add_argument("--num-microbatches", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-thd", action="store_true")
    parser.add_argument("--same-data-across-dp", action="store_true")
    parser.add_argument("--no-optimizer", action="store_true")
    parser.add_argument("--skip-load-hf-weights", action="store_true")
    parser.add_argument("--skip-optimizer-build", action="store_true")
    parser.add_argument("--keep-experts", type=int, default=None)
    parser.add_argument("--truncate-layers", type=int, default=None)
    parser.add_argument("--disable-mtp", action="store_true")
    parser.add_argument("--optimizer-lr", type=float, default=1e-4)
    parser.add_argument("--optimizer-weight-decay", type=float, default=0.1)
    parser.add_argument("--optimizer-clip-grad", type=float, default=1.0)
    parser.add_argument("--override-ddp-json", default="{}")
    parser.add_argument("--override-transformer-json", default="{}")
    parser.add_argument("--override-optimizer-json", default="{}")
    parser.add_argument("--impl-cfg-json", default="{}")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--skip-weight-hash", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--activation-probes-json", default="{}")
    parser.add_argument("--verify-resume-path", default=None)


def _activation_probe_names(raw: str, backend: str) -> list[str]:
    value = json.loads(raw)
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        selected = value.get(backend, [])
        if not isinstance(selected, list):
            raise ValueError(
                f"activation probe list for {backend!r} must be a JSON list."
            )
        return [str(item) for item in selected]
    raise ValueError(
        "activation_probes_json must be a JSON list or backend-to-list mapping."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser(
        "run", help="run one backend and write a correctness artifact"
    )
    _add_run_args(run_p)

    cmp_p = sub.add_parser("compare", help="strictly compare two correctness artifacts")
    cmp_p.add_argument("baseline")
    cmp_p.add_argument("candidate")
    cmp_p.add_argument("--output-json", default=None)
    cmp_p.add_argument("--fail-on-mismatch", action="store_true")
    cmp_p.add_argument("--loss-atol", type=float, default=0.0)
    cmp_p.add_argument("--loss-rtol", type=float, default=0.0)
    cmp_p.add_argument("--grad-atol", type=float, default=0.0)
    cmp_p.add_argument("--grad-rtol", type=float, default=0.0)
    cmp_p.add_argument("--tensor-atol", type=float, default=0.0)
    cmp_p.add_argument("--tensor-rtol", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ns = _parser().parse_args(argv)
    if ns.command == "compare":
        result = compare_correctness_artifacts(
            load_result_artifact(ns.baseline),
            load_result_artifact(ns.candidate),
            loss_atol=ns.loss_atol,
            loss_rtol=ns.loss_rtol,
            grad_atol=ns.grad_atol,
            grad_rtol=ns.grad_rtol,
            tensor_atol=ns.tensor_atol,
            tensor_rtol=ns.tensor_rtol,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text, flush=True)
        if ns.output_json:
            Path(ns.output_json).write_text(text + "\n", encoding="utf-8")
        if ns.fail_on_mismatch and not result["passed"]:
            raise SystemExit(1)
        return result

    cfg = BenchCliConfig(
        **{
            k: v
            for k, v in vars(ns).items()
            if k in BenchCliConfig.__dataclass_fields__
        }
    )
    artifact = run_backend(
        cfg,
        hash_weights=not ns.skip_weight_hash,
        activation_probe_names=_activation_probe_names(
            ns.activation_probes_json, cfg.backend
        ),
        verify_resume_path=ns.verify_resume_path,
        performance_warmup_steps=ns.performance_warmup_steps,
        performance_steps=ns.performance_steps,
    )
    if _distributed_rank() == 0:
        text = json.dumps(artifact, indent=2, sort_keys=True)
        print(text, flush=True)
        output_path = Path(ns.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":
    main()
