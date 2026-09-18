"""Real checkpoint composition gate; this does not replace rollout/runtime tests."""

import os
from pathlib import Path

import pytest
import torch


def test_protocol_full_recompute_wraps_every_local_layer(monkeypatch):
    from types import SimpleNamespace

    from megatron.lite.model.nemotron_h import protocol

    class FakeModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.layers = torch.nn.ModuleDict(
                {"0": torch.nn.Identity(), "1": torch.nn.Identity()}
            )

    ps = SimpleNamespace(pp_rank=0, pp_size=1)
    calls = []
    monkeypatch.setattr(protocol, "init_parallel", lambda _: ps)
    monkeypatch.setattr(protocol, "NemotronModel", FakeModel)
    monkeypatch.setattr(
        protocol,
        "apply_recompute",
        lambda layers, spec, module_map: calls.append(
            (list(layers), spec, module_map)
        ),
    )
    bundle = protocol.build_model(
        SimpleNamespace(num_hidden_layers=2),
        impl_cfg=protocol.ImplConfig(optimizer=None, recompute="full"),
    )
    assert calls == [
        (list(bundle.chunks[0].layers.values()), ["full"], protocol.MODULE_MAP)
    ]


@pytest.mark.gpus(1)
def test_full_recompute_preserves_nemotron_layer_forward_and_backward():
    from types import SimpleNamespace

    from megatron.lite.model.nemotron_h.model import NemotronModel
    from megatron.lite.primitive.recompute import apply_recompute

    class Layer(torch.nn.Module):
        def forward(self, hidden, residual, meta):
            assert meta is sentinel
            residual = hidden if residual is None else residual
            return hidden.square(), residual

    sentinel = object()
    model = NemotronModel.__new__(NemotronModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=3)
    model.pre_process = True
    model.post_process = False
    model.embeddings = torch.nn.Embedding(5, 3)
    model.layers = torch.nn.ModuleDict({"0": Layer()})
    apply_recompute(model.layers.values(), ["full"], {})

    output = model(torch.tensor([1, 2]), meta=sentinel)
    expected_hidden = model.embeddings(torch.tensor([1, 2]))
    torch.testing.assert_close(output[:, 0, :3], expected_hidden.square())
    torch.testing.assert_close(output[:, 0, 3:], expected_hidden)
    output.sum().backward()
    assert model.embeddings.weight.grad is not None


@pytest.mark.parametrize("mask", [[1, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0] * 6])
def test_cp_loss_unequal_valid_counts_matches_unsharded_gradient(mask):
    from megatron.lite.model.nemotron_h.protocol import _token_mean_loss

    scores = -torch.arange(1.0, 7.0, requires_grad=True)
    weights = torch.tensor(mask, dtype=torch.float32)
    reference = -(scores * weights).sum() / weights.sum().clamp_min(1)
    # Reproduce Megatron's CP-averaged parameter gradient, including an empty rank.
    actual = (
        sum(
            _token_mean_loss(local, local_mask, weights, 2)
            for local, local_mask in zip(scores.chunk(2), weights.chunk(2), strict=True)
        )
        / 2
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual, scores, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(reference, scores)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("ep_rank", [0, 1])
def test_export_stacked_experts_preserves_hf_layout(ep_rank):
    """Export views retain values and use local IDs before EP gathering."""
    from types import SimpleNamespace

    from megatron.lite.model.nemotron_h.checkpoint import NemotronExport
    from megatron.lite.primitive.ckpt.hf_weights import export_hf_weights

    model = torch.nn.Module()
    model.config = SimpleNamespace(n_routed_experts=4)
    model.ps = SimpleNamespace(ep_size=2, ep_rank=ep_rank)
    model.layers = torch.nn.ModuleDict({"7": torch.nn.Module()})
    layer = model.layers["7"]
    layer.mixer = torch.nn.Module()
    layer.mixer.experts = torch.nn.Module()
    for name in ("up_proj", "down_proj"):
        layer.mixer.experts.register_parameter(
            name, torch.nn.Parameter(torch.arange(12.0).reshape(2, 2, 3) + ep_rank * 20)
        )
    layer.mixer.register_buffer("e_score_correction_bias", torch.arange(4.0))
    spec = NemotronExport(model.config)
    views = dict(spec.iter_export_tensors(model))
    for local in range(2):
        for projection in ("up_proj", "down_proj"):
            name = f"backbone.layers.7.mixer.experts.{local}.{projection}.weight"
            torch.testing.assert_close(
                views[name],
                getattr(layer.mixer.experts, projection)[local],
                rtol=0,
                atol=0,
            )
            assert spec.export_expert_name(name, ep_rank * 2 + local).endswith(
                f"experts.{ep_rank * 2 + local}.{projection}.weight"
            )
    # Exercise the actual framework exporter locally; distributed gather has
    # a separate gate. Persistent router buffers must not disappear.
    ps = SimpleNamespace(pp_size=1, tp_size=1, ep_size=1, etp_size=1, ep_group=None)
    exported = dict(export_hf_weights(model, spec, ps, cpu=True))
    assert exported.keys() == views.keys()
    for name, value in views.items():
        torch.testing.assert_close(exported[name], value, rtol=0, atol=0)


@pytest.mark.gpus(1)
def test_real_seven_layer_model_logits_match_frozen_hf_alignment():
    if not torch.cuda.is_available() or "NEMOTRON_TEST_MODEL" not in os.environ:
        pytest.skip("CUDA and NEMOTRON_TEST_MODEL required")
    import nemotron_h_reference as oracle
    from megatron.lite.model.nemotron_h.checkpoint import load_hf_weights
    from megatron.lite.model.nemotron_h.config import NemotronHConfig
    from megatron.lite.model.nemotron_h.mamba import SSMMeta
    from megatron.lite.model.nemotron_h.model import NemotronModel
    from megatron.lite.primitive.parallel import ParallelState
    from transformers import AutoConfig
    from transformers.models.nemotron_h import modeling_nemotron_h as hf

    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    root = Path(os.environ["NEMOTRON_TEST_MODEL"])
    init_batch_invariance()
    config = NemotronHConfig.from_hf(root)
    config.layers_block_type = config.layers_block_type[:7]
    candidate = NemotronModel(config, ParallelState(), device="cuda").eval()
    load_hf_weights(candidate, root)
    hf_config = AutoConfig.from_pretrained(root, local_files_only=True)
    hf_config.layers_block_type = hf_config.layers_block_type[:7]
    hf_config.num_hidden_layers = 7
    hf_config.num_nextn_predict_layers = 0
    hf_config.time_step_limit = (0.0, float("inf"))
    hf_config._attn_implementation = "eager"
    hf_config._experts_implementation = "eager"
    with torch.device("meta"):
        reference = hf.NemotronHForCausalLM(hf_config).to(dtype=torch.bfloat16)
    # Share immutable loaded tensors; this gate tests composition, not HF conversion.
    reference.load_state_dict(
        {
            (key if key.startswith("lm_head.") else f"model.{key}"): value
            for key, value in candidate.state_dict().items()
        },
        strict=True,
        assign=True,
    )
    reference.eval()
    originals = {
        name: getattr(hf, name)
        for name in (
            "causal_conv1d_fn",
            "mamba2_chunk_scan",
            "eager_attention_forward",
        )
    }
    hidden_candidate, hidden_reference = {}, {}

    def save_hidden(target, index):
        def hook(module, inputs, output):
            target[index] = output

        return hook

    for index, layer in candidate.layers.items():
        layer.register_forward_hook(save_hidden(hidden_candidate, int(index)))
    for index, layer in enumerate(reference.model.layers):
        layer.register_forward_hook(save_hidden(hidden_reference, index))
    try:
        oracle.install_training_alignment(reference)
        torch.manual_seed(84)
        ids = torch.randint(config.vocab_size, (36,), device="cuda")
        positions = torch.cat((torch.arange(17), torch.arange(19))).to("cuda")[None]
        with torch.no_grad():
            actual = candidate(ids, meta=SSMMeta((0, 17, 36)))
            expected = reference(
                input_ids=ids[None], position_ids=positions, use_cache=False
            ).logits[0]
        for index in range(7):
            for stream in range(2):
                torch.testing.assert_close(
                    hidden_candidate[index][stream],
                    hidden_reference[index][stream][0],
                    rtol=0,
                    atol=0,
                    msg=f"layer={index} stream={stream}",
                )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        from megatron.lite.model.nemotron_h.protocol import forward_step
        from megatron.lite.runtime.contracts.data import PackedBatch

        batch = PackedBatch(ids, ids, torch.tensor([17, 19], device="cuda"))
        with torch.no_grad():
            scored = forward_step(candidate, batch)
            targets = torch.cat((ids[:17].roll(-1), ids[17:].roll(-1)))
            targets[[16, 35]] = 0
            expected_scores = oracle.token_logprobs(expected, targets[:, None]).reshape(
                1, -1
            )
        torch.testing.assert_close(scored["log_probs"], expected_scores, rtol=0, atol=0)
    finally:
        for name, function in originals.items():
            setattr(hf, name, function)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("optimizer", [None, "dist_opt"])
def test_real_seven_layer_native_runtime_forward(tmp_path, optimizer):
    if not torch.cuda.is_available() or "NEMOTRON_TEST_MODEL" not in os.environ:
        pytest.skip("CUDA and NEMOTRON_TEST_MODEL required")
    from megatron.lite.model.nemotron_h.protocol import forward_step
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
    from megatron.lite.runtime.contracts.data import PackedBatch

    def first_seven(config):
        config.layers_block_type = config.layers_block_type[:7]
        return config

    torch.cuda.set_device(0)
    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'runtime-init'}", rank=0, world_size=1
    )
    runtime = None
    handle = None
    try:
        root = os.environ["NEMOTRON_TEST_MODEL"]
        config = MegatronLiteConfig(
            model_name="nemotron_h",
            hf_path=root,
            impl_cfg={"optimizer": optimizer},
            model_config_hook=first_seven,
        )
        runtime = MegatronLiteRuntime(root, config)
        handle = runtime.build_model()
        torch.manual_seed(85)
        ids = torch.randint(
            handle._extras["model_cfg"].vocab_size, (36,), device="cuda"
        )
        batch = PackedBatch(ids, ids, torch.tensor([17, 19], device="cuda"))
        with torch.no_grad():
            expected = forward_step(handle._model, batch)
        result = runtime.forward_backward(handle, [batch], None, forward_only=True)
        torch.testing.assert_close(
            result.model_output.log_probs, expected["log_probs"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.model_output.loss, expected["loss"], rtol=0, atol=0
        )
        if optimizer is None:
            assert handle._optimizer is None
        else:
            model = handle._model
            while hasattr(model, "module"):
                model = model.module
            before = model.lm_head.weight[ids[1]].detach().clone()
            runtime.zero_grad(handle)
            trained = runtime.forward_backward(
                handle, [batch], None, forward_only=False
            )
            assert torch.isfinite(trained.model_output.loss)
            success, norm, _ = runtime.optimizer_step(handle)
            assert success and norm > 0 and torch.isfinite(torch.tensor(norm))
            assert not torch.equal(before, model.lm_head.weight[ids[1]])
            checkpoint_dir = os.environ.get("NEMOTRON_TEST_CHECKPOINT_DIR")
            if checkpoint_dir:
                with torch.no_grad():
                    after_one = forward_step(handle._model, batch)["log_probs"].clone()
                runtime.save_checkpoint(handle, checkpoint_dir, step=1)

                def update_again():
                    runtime.zero_grad(handle)
                    runtime.forward_backward(handle, [batch], None, forward_only=False)
                    updated, grad_norm, _ = runtime.optimizer_step(handle)
                    assert updated and grad_norm > 0
                    with torch.no_grad():
                        return forward_step(handle._model, batch)["log_probs"].clone()

                after_two = update_again()
                loaded_step = runtime.load_checkpoint(
                    handle, str(Path(checkpoint_dir) / "step_1")
                )
                assert loaded_step == 1
                with torch.no_grad():
                    restored = forward_step(handle._model, batch)["log_probs"]
                torch.testing.assert_close(restored, after_one, rtol=0, atol=0)
                torch.testing.assert_close(update_again(), after_two, rtol=0, atol=0)
                print(
                    "NEMOTRON_NATIVE_CHECKPOINT_STEP1_RESTORE_AND_STEP2_EXACT",
                    flush=True,
                )
    finally:
        if runtime is not None and handle is not None:
            runtime.close(handle)
        if optimizer is not None:
            from megatron.core import parallel_state

            parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def _distributed_runtime_worker(rank, init_file, mode, layer_count=7, optimizer=None):
    from megatron.lite.model.nemotron_h.checkpoint import (
        hf_tensor_views,
        load_hf_weights,
    )
    from megatron.lite.model.nemotron_h.model import NemotronModel
    from megatron.lite.model.nemotron_h.protocol import (
        forward_step,
        unpack_forward_output,
    )
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
    from megatron.lite.runtime.contracts import ParallelConfig
    from megatron.lite.runtime.contracts.data import PackedBatch

    combined = mode == "combined"
    device = int(os.environ["LOCAL_RANK"]) if combined else rank
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        "nccl",
        init_method="env://" if combined else f"file://{init_file}",
        rank=rank,
        world_size=8 if combined else 2,
    )
    handle = None
    try:
        root = os.environ["NEMOTRON_TEST_MODEL"]

        def select_layers(config):
            assert len(config.layers_block_type) >= layer_count
            config.layers_block_type = config.layers_block_type[:layer_count]
            return config

        parallel = {
            "cp_ep": ParallelConfig(cp=2, ep=2),
            "pp": ParallelConfig(pp=2),
            "combined": ParallelConfig(pp=2, cp=2, ep=4),
        }[mode]
        config = MegatronLiteConfig(
            model_name="nemotron_h",
            hf_path=root,
            parallel=parallel,
            impl_cfg={"optimizer": optimizer},
            model_config_hook=select_layers,
        )
        runtime = MegatronLiteRuntime(root, config)
        handle = runtime.build_model()
        model_cfg = handle._extras["model_cfg"]
        reference = NemotronModel(
            model_cfg, ParallelState(), device=torch.device("cuda", device)
        )
        load_hf_weights(reference, root)
        expected_weights = dict(hf_tensor_views(reference))
        seen = set()
        for name, tensor in runtime.export_weights(
            handle, buffer_max_size_bytes=16 * 1024**2
        ):
            assert name not in seen
            torch.testing.assert_close(tensor, expected_weights[name], rtol=0, atol=0)
            seen.add(name)
        assert seen == expected_weights.keys()
        print(
            f"NEMOTRON_NATIVE_EXPORT_EXACT rank={rank} tensors={len(seen)}", flush=True
        )
        dp_rank = handle._parallel_state.dp_rank
        torch.manual_seed(86 + dp_rank)
        lengths = [17 + dp_rank, 19 + 2 * dp_rank]
        ids = torch.randint(model_cfg.vocab_size, (sum(lengths),), device="cuda")
        batch = PackedBatch(ids, ids, torch.tensor(lengths, device="cuda"))
        captured = []
        original_step = handle._extras["forward_step"]

        def capture(model, batch):
            output = original_step(model, batch)
            if "log_probs" in output:
                captured.append(output["log_probs"])
            return output

        handle._extras["forward_step"] = capture
        with torch.no_grad():
            expected = forward_step(reference, batch)["log_probs"].reshape(-1)
            runtime.forward_backward(handle, [batch], None, forward_only=True)
            if handle._parallel_state.pp_is_last:
                actual = unpack_forward_output(
                    handle._model, batch, captured[0]
                ).values()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                assert not captured
        print(f"NEMOTRON_NATIVE_FORWARD_EXACT rank={rank} mode={mode}", flush=True)
        if optimizer is not None:
            captured.clear()
            runtime.zero_grad(handle)
            runtime.forward_backward(handle, [batch], None, forward_only=False)
            success, norm, _ = runtime.optimizer_step(handle)
            assert success and norm > 0 and torch.isfinite(torch.tensor(norm))
            changed = 0
            seen.clear()
            for name, tensor in runtime.export_weights(
                handle, buffer_max_size_bytes=16 * 1024**2
            ):
                assert name not in seen
                assert tensor.shape == expected_weights[name].shape
                assert tensor.dtype == expected_weights[name].dtype
                assert torch.isfinite(tensor).all()
                changed += int(not torch.equal(tensor, expected_weights[name]))
                seen.add(name)
            assert seen == expected_weights.keys() and changed > 0
            print(
                f"NEMOTRON_NATIVE_DISTOPT_UPDATED rank={rank} "
                f"grad_norm={norm} changed_tensors={changed}",
                flush=True,
            )
            destination = os.environ.get("NEMOTRON_UPDATED_EXPORT")
            if destination:
                import json
                import shutil

                from megatron.lite.primitive.ckpt.hf_weights import (
                    stream_export_to_shards,
                )

                target = Path(destination)
                if rank == 0:
                    assert not target.exists(), "Do not overwrite an earlier export"
                torch.distributed.barrier()
                print(
                    f"NEMOTRON_NATIVE_HF_SAVE_BEGIN rank={rank} path={target}",
                    flush=True,
                )
                stream_export_to_shards(
                    runtime.export_weights(handle, cpu=True, rank0_only=True),
                    destination,
                    shard_size_bytes=1024**3,
                )
                if rank == 0:
                    hf_config = json.loads((Path(root) / "config.json").read_text())
                    hf_config["layers_block_type"] = model_cfg.layers_block_type
                    hf_config["num_hidden_layers"] = layer_count
                    (target / "config.json").write_text(json.dumps(hf_config, indent=2))
                    for filename in (
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                        "generation_config.json",
                        "LICENSE",
                    ):
                        if (Path(root) / filename).is_file():
                            shutil.copyfile(Path(root) / filename, target / filename)
                torch.distributed.barrier()
                print(
                    f"NEMOTRON_NATIVE_UPDATED_HF_SAVED rank={rank} path={target}",
                    flush=True,
                )
    finally:
        if handle is not None:
            runtime.close(handle)
        if optimizer is not None:
            from megatron.core import parallel_state

            parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


@pytest.mark.gpus(2)
@pytest.mark.parametrize("mode", ["cp_ep", "pp"])
def test_real_seven_layer_distributed_runtime(tmp_path, mode):
    if torch.cuda.device_count() < 2 or "NEMOTRON_TEST_MODEL" not in os.environ:
        pytest.skip("Two GPUs and NEMOTRON_TEST_MODEL required")
    torch.multiprocessing.spawn(
        _distributed_runtime_worker,
        args=(str(tmp_path / "distributed-init"), mode),
        nprocs=2,
    )


@pytest.mark.gpus(8)
@pytest.mark.parametrize("layer_count", [7, 52])
def test_real_combined_runtime(layer_count):
    if os.environ.get("WORLD_SIZE") != "8" or "NEMOTRON_TEST_MODEL" not in os.environ:
        pytest.skip("Launch with torchrun on eight GPUs and NEMOTRON_TEST_MODEL")
    _distributed_runtime_worker(int(os.environ["RANK"]), None, "combined", layer_count)


@pytest.mark.gpus(8)
@pytest.mark.parametrize("layer_count", [8, 52])
def test_real_combined_distopt_update(layer_count):
    if os.environ.get("WORLD_SIZE") != "8" or "NEMOTRON_TEST_MODEL" not in os.environ:
        pytest.skip("Launch with torchrun on eight GPUs and NEMOTRON_TEST_MODEL")
    _distributed_runtime_worker(
        int(os.environ["RANK"]), None, "combined", layer_count, "dist_opt"
    )


@pytest.mark.gpus(8)
def test_native_full_model_frozen_rollout_logprobs():
    """Actual rollout tokens/logprobs, not a native-model self comparison."""
    import json

    if os.environ.get("WORLD_SIZE") != "8" or "NEMOTRON_ROLLOUT" not in os.environ:
        pytest.skip("Eight-rank torchrun and a frozen same-weight rollout required")
    from megatron.lite.model.nemotron_h.protocol import unpack_forward_output
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
    from megatron.lite.runtime.contracts import ParallelConfig
    from megatron.lite.runtime.contracts.data import PackedBatch
    from megatron.lite.runtime.contracts.loss import LossContext

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    handle = None
    try:
        root = os.environ["NEMOTRON_TEST_MODEL"]
        runtime = MegatronLiteRuntime(
            root,
            MegatronLiteConfig(
                model_name="nemotron_h",
                hf_path=root,
                parallel=ParallelConfig(pp=2, cp=2, ep=4),
                impl_cfg={"optimizer": None},
            ),
        )
        handle = runtime.build_model()
        rows = json.loads(Path(os.environ["NEMOTRON_ROLLOUT"]).read_text())
        dp = handle._parallel_state.dp_rank
        selected = [rows[dp], rows[dp + 2]]
        captured = []
        original = handle._extras["forward_step"]

        def capture(model, batch):
            output = original(model, batch)
            if "log_probs" in output:
                assert torch.isfinite(output["entropy"]).all()
                captured.append(output["log_probs"])
            return output

        handle._extras["forward_step"] = capture
        for case in ([selected[0]], selected, selected[::-1]):
            lengths = [len(row["prompt"]) + len(row["tokens"]) for row in case]
            ids = torch.tensor(
                [token for row in case for token in row["prompt"] + row["tokens"]],
                device="cuda",
            )
            batch = PackedBatch(ids, ids, torch.tensor(lengths, device="cuda"))
            captured.clear()
            with torch.no_grad():
                runtime.forward_backward(
                    handle,
                    [(batch, LossContext(calculate_entropy=True))],
                    None,
                    forward_only=True,
                )
                if handle._parallel_state.pp_is_last:
                    values = unpack_forward_output(
                        handle._model, batch, captured[0]
                    ).unbind()
                    for row, scores in zip(case, values, strict=True):
                        actual = scores.reshape(-1)[len(row["prompt"]) - 1 : -1]
                        expected = torch.tensor(row["rollout_logprobs"], device="cuda")
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            print(
                f"NEMOTRON_NATIVE_ROLLOUT_EXACT rank={torch.distributed.get_rank()} "
                f"lengths={lengths} scored={handle._parallel_state.pp_is_last}",
                flush=True,
            )
    finally:
        if handle is not None:
            runtime.close(handle)
        torch.distributed.destroy_process_group()
