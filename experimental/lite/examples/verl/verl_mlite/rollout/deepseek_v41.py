# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native DS4.1 row/quantized receiver, selected with worker_extension_cls.

Requires vLLM's layerwise reload metadata recorded before initial processing.
Engram tables remain resident: only each shard's intersecting rows are copied.
"""
from contextlib import contextmanager
from copy import copy
from functools import wraps

import torch
from megatron.lite.model.deepseek_v41.lite.resync import decode_transport
from megatron.lite.primitive.ckpt.row_stream import RowChunk, RowReceiver


def set_mxfp4_load_numel(model):
    """Use native MXFP4 creation for checkpoint-sized reload thresholds.

    Reuse the same layout as cold loading/export, before metadata capture.
    Only the disposable method copy is mutated; no kernel tensors are changed.
    """
    for layer in model.modules():
        method = getattr(layer, 'quant_method', None)
        if type(method).__name__ != 'Mxfp4MoEMethod':
            continue
        config = layer.moe_config
        tp = config.moe_parallel_config.tp_size
        intermediate = config.intermediate_size
        if config.tp_shard_with_padding or intermediate % (tp * 32):
            raise NotImplementedError(
                'DS4.1 MXFP4 reload requires evenly sharded group32 checkpoints'
            )
        template = torch.nn.Module()
        with torch.device('meta'):
            copy(method).create_weights(
                template,
                num_experts=layer.w13_weight.shape[0],
                hidden_size=config.hidden_dim_unpadded,
                intermediate_size_per_partition=intermediate // tp,
                params_dtype=layer.w13_weight.dtype,
                weight_loader=layer.w13_weight.weight_loader,
            )
        for name, parameter in template.named_parameters(recurse=False):
            target = layer.get_parameter(name)
            if parameter.numel() > target.numel():
                raise ValueError('DS4.1 MXFP4 checkpoint exceeds padded parameter')
            target.weight_loader_numel = parameter.numel()


class StagingBudget:
    """Count actual retained input storage, including aliases, across all layers.

    Separate from exporter/transport workspace and resident kernel weights.
    Check before cloning, so a missing completion cannot grow without bound.
    """

    def __init__(self, infos, budget_bytes):
        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise ValueError('Invalid DS4.1 receiver staging budget')
        self.infos, self.budget_bytes = infos, budget_bytes
        self.current_bytes = self.peak_bytes = 0

    def refresh(self):
        storages = {}

        def visit(value):
            if isinstance(value, torch.Tensor) and value.device.type != 'meta':
                storage = value.untyped_storage()
                storages[(value.device, storage.data_ptr())] = storage.nbytes()
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    visit(child)

        for info in self.infos:
            for _, bound in info.loaded_weights:
                for name, value in bound.arguments.items():
                    if name != 'param':
                        visit(value)
        self.current_bytes = sum(storages.values())
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        return self.current_bytes

    def check(self, incoming_bytes):
        total = self.refresh() + incoming_bytes
        if total > self.budget_bytes:
            raise RuntimeError(
                f'DS4.1 receiver staging {total} exceeds budget {self.budget_bytes}'
            )
        self.peak_bytes = max(self.peak_bytes, total)


class MegaAttnReload:
    """Apply native checkpoint permutations before per-projection packing.

    Cold load finalizes MegaAttn inside model.load_weights, before the loader
    processes quant methods. Layerwise reload instead processes each completed
    projection immediately. Run the same native permutations at that boundary;
    the model-level finalizer must then skip these already fused weights.
    """

    def __init__(self, model):
        self.groups, self.methods = [], []
        for attention in model.modules():
            if type(attention).__name__ != 'DeepseekV4MegaAttnAttention':
                continue
            from vllm.models.deepseek_v41.common.ops import fused_layout

            completed = set()
            self.groups.append((attention, completed))
            attention._fused_layouts_ready = False
            for name, permute, heads in (
                ('wq_b', fused_layout.permute_wq_b_, attention.n_local_heads),
                (
                    'wo_a',
                    fused_layout.permute_wo_a_,
                    attention.n_local_heads // attention.n_local_groups,
                ),
            ):
                projection = getattr(attention, name)
                # Isolate hooks even if two projections share a quant method.
                method = copy(projection.quant_method)
                method.__dict__.pop('process_weights_after_loading', None)
                process = method.process_weights_after_loading

                def before_pack(
                    layer,
                    process=process,
                    permute=permute,
                    heads=heads,
                    name=name,
                    attention=attention,
                    completed=completed,
                ):
                    if name in completed:
                        raise RuntimeError('DS4.1 MegaAttn projection processed twice')
                    permute(layer.weight.data, layer.weight_scale.data, heads)
                    process(layer)
                    completed.add(name)
                    attention._fused_layouts_ready = completed == {'wq_b', 'wo_a'}

                method.process_weights_after_loading = before_pack
                projection.quant_method = method
                self.methods.append(method)

    def finish(self):
        if any(done != {'wq_b', 'wo_a'} for _, done in self.groups):
            raise ValueError('DS4.1 resync incomplete MegaAttn projections')

    def restore(self):
        for method in self.methods:
            method.__dict__.pop('process_weights_after_loading', None)


@contextmanager
def _without_tables(model):
    removed = []
    for parent in list(model.modules()):
        for name, child in list(parent._modules.items()):
            if child is not None and any(
                hasattr(p, 'engram_vocab_start')
                for p in child.parameters(recurse=False)
            ):
                removed.append((parent, name, child))
                del parent._modules[name]
    try:
        yield
    finally:
        for parent, name, child in removed:
            parent._modules[name] = child


def install_reload_metadata_hook():
    """Install in the rollout worker before its loader creates the model."""
    from vllm.model_executor.model_loader import reload
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    original = BaseModelLoader.create_model
    if getattr(original, '_ds41_metadata_hook', False):
        return

    @wraps(original)
    def create(self, vllm_config, model_config, prefix=''):
        model = original(self, vllm_config, model_config, prefix)
        if getattr(model_config.hf_config, 'model_type', '') == 'deepseek_v41':
            set_mxfp4_load_numel(model)
            with _without_tables(model):
                reload.record_metadata_for_reloading(model)
            model._ds41_reload_metadata = True
        return model

    create._ds41_metadata_hook = True
    BaseModelLoader.create_model = create


class ResyncReceiver:
    """One complete generation; bucket boundaries do not finalize the model."""

    def __init__(self, model, model_config, staging_budget_bytes=16 * 1024**3):
        from vllm.model_executor.model_loader import reload
        from vllm.model_executor.model_loader.reload import layerwise

        if not getattr(model, '_ds41_reload_metadata', False):
            raise RuntimeError(
                'DS4.1 resync requires the metadata hook before model load'
            )
        if getattr(model_config, 'cpu_offload_gb', 0):
            raise NotImplementedError('DS4.1 resync CPU offload is not supported')
        if any(getattr(m, 'use_mega_moe', False) for m in model.modules()):
            raise NotImplementedError(
                'DS4.1 resync currently requires FusedMoE, not MegaMoE'
            )
        self.model, self.model_config = model, model_config
        self.staging = StagingBudget(
            [layerwise.get_layerwise_info(m) for m in model.modules()],
            staging_budget_bytes,
        )
        self.rows, self.hooks = {}, []
        self.expected_tables = {
            n
            for n, p in model.named_parameters()
            if n.endswith('.weight') and hasattr(p, 'engram_vocab_start')
        }
        self.received_tables = set()
        self.finished = self.ended = False
        # load_weights invokes these model hooks per bucket. Defer all of them
        # until weight AND scale tensors for the entire generation have arrived.
        for module in model.modules():
            if hasattr(module, '_fused_layouts_ready'):
                module._fused_layouts_ready = False
            if hasattr(module, 'process_weights_after_loading') and (
                type(module).__name__
                in ('DeepseekV41LLMForCausalLM', 'DeepseekV41ForCausalLM')
                or module is model
            ):
                self.hooks.append((module, module.process_weights_after_loading))
                module.process_weights_after_loading = lambda: None
        with _without_tables(model):
            reload.initialize_layerwise_reload(model)
        self.mega_attn = MegaAttnReload(model)

    @torch.no_grad()
    def receive(self, weights):
        if self.finished:
            raise RuntimeError('DS4.1 resync generation already finalized')
        for name, tensor in weights:
            if self.ended:
                raise ValueError('DS4.1 tensors after generation terminator')
            item = decode_transport(name, tensor)
            if item is None:
                self.ended = True
                continue
            if isinstance(item, RowChunk):
                if item.name not in self.rows:
                    mapped = list(
                        self.model.hf_to_vllm_mapper.apply([(item.name, item.weight)])
                    )
                    if len(mapped) != 1:
                        raise ValueError(
                            'DS4.1 Engram row name must map to one parameter'
                        )
                    self.received_tables.add(mapped[0][0])
                    weight = self.model.get_parameter(mapped[0][0])
                    scale = self.model.get_parameter(
                        mapped[0][0][:-6] + 'weight_scale_inv'
                    )
                    self.rows[item.name] = RowReceiver(
                        item.name,
                        item.total_rows,
                        weight.engram_vocab_start,
                        weight,
                        scale,
                    )
                self.rows[item.name].copy(item)
            else:
                name, tensor = item
                # The upstream layerwise loader may retain a tensor until its
                # module is complete; IPC storage is borrowed only until return.
                if tensor.dtype == torch.int8:
                    tensor = tensor.view(torch.uint8)
                try:
                    self.staging.check(tensor.numel() * tensor.element_size())
                    self.model.load_weights([(name, tensor.clone())])
                    self.staging.refresh()
                except BaseException:
                    self.abort()
                    raise

    def abort(self):
        """Release unconsumed inputs and restore hooks after a failed refit.

        The partially updated generation must be discarded by the caller.
        Do not repack an incomplete layer during error cleanup.
        """
        from vllm.model_executor.model_loader.reload import layerwise

        for layer in self.model.modules():
            info = layerwise.get_layerwise_info(layer)
            if info.can_load() and info.kernel_tensors is not None:
                layerwise._place_kernel_tensors(layer, info)
            info.reset()
            layerwise.LOADING_LAYERS.discard(layer)
        for module, hook in self.hooks:
            module.process_weights_after_loading = hook
        self.mega_attn.restore()
        if hasattr(self.model, '_original_do_torchao_reload'):
            self.model._do_torchao_reload = self.model._original_do_torchao_reload
        self.staging.refresh()
        self.finished = True

    def finish(self):
        from vllm.model_executor.model_loader import reload

        if self.finished:
            raise RuntimeError('DS4.1 resync generation already finalized')
        if not self.ended:
            raise ValueError('DS4.1 resync incomplete generation')
        if self.received_tables != self.expected_tables:
            raise ValueError('DS4.1 resync missing Engram tables')
        for receiver in self.rows.values():
            receiver.finish()
        try:
            with _without_tables(self.model):
                reload.finalize_layerwise_reload(self.model, self.model_config)
            self.mega_attn.finish()
            self.mega_attn.restore()
            for module, hook in self.hooks:
                module.process_weights_after_loading = hook
            if hasattr(self.model, '_weights_finalized'):
                self.model._weights_finalized = False
            self.model.process_weights_after_loading()
            self.staging.refresh()
            self.finished = True
        except BaseException:
            self.abort()
            raise


def worker_extension():
    import verl.workers.rollout.vllm_rollout.utils as utils
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    class DeepseekV41WorkerExtension(utils.vLLMColocateWorkerExtension):
        def __new__(cls, **kwargs):
            install_reload_metadata_hook()
            return super().__new__(cls, **kwargs)

        def update_weights_from_ipc(
            self, peft_config=None, base_sync_done=False, use_shm=False
        ):
            if (
                peft_config
                or self.model_runner.vllm_config.speculative_config is not None
            ):
                raise NotImplementedError(
                    'DS4.1 resync supports base weights without a drafter'
                )
            receiver = ResyncReceiver(
                self.model_runner.model, self.model_runner.vllm_config.model_config
            )
            transport = bucketed_weight_transfer.BucketedWeightReceiver(
                zmq_handle=self._get_zmq_handle(), device=self.device, use_shm=use_shm
            )
            transport.receive_weights(
                on_bucket_received=lambda weights, is_last: receiver.receive(weights)
            )
            receiver.finish()

    return DeepseekV41WorkerExtension


# Loaded lazily by vLLM; importing the byte/row consumer itself needs no verl.
def __getattr__(name):
    if name == 'DeepseekV41WorkerExtension':
        cls = worker_extension()
        cls.__qualname__ = name
        globals()[name] = cls
        return cls
    raise AttributeError(name)
