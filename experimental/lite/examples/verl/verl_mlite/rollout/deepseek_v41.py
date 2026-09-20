# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native DS4.1 row/quantized receiver, selected with worker_extension_cls.

Requires vLLM's layerwise reload metadata recorded before initial processing.
Engram tables remain resident: only each shard's intersecting rows are copied.
"""
from contextlib import contextmanager
from functools import wraps

import torch
from megatron.lite.model.deepseek_v41.lite.resync import decode_transport
from megatron.lite.primitive.ckpt.row_stream import RowChunk, RowReceiver


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
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.model_loader.reload import record_metadata_for_reloading

    original = BaseModelLoader.create_model
    if getattr(original, '_ds41_metadata_hook', False):
        return

    @wraps(original)
    def create(self, vllm_config, model_config, prefix=''):
        model = original(self, vllm_config, model_config, prefix)
        if getattr(model_config.hf_config, 'model_type', '') == 'deepseek_v41':
            with _without_tables(model):
                record_metadata_for_reloading(model)
            model._ds41_reload_metadata = True
        return model

    create._ds41_metadata_hook = True
    BaseModelLoader.create_model = create


class ResyncReceiver:
    """One complete generation; bucket boundaries do not finalize the model."""

    def __init__(self, model, model_config):
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

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
            initialize_layerwise_reload(model)

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
                self.model.load_weights([(name, tensor.clone())])

    def finish(self):
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

        if self.finished:
            raise RuntimeError('DS4.1 resync generation already finalized')
        if not self.ended:
            raise ValueError('DS4.1 resync incomplete generation')
        if self.received_tables != self.expected_tables:
            raise ValueError('DS4.1 resync missing Engram tables')
        for receiver in self.rows.values():
            receiver.finish()
        with _without_tables(self.model):
            finalize_layerwise_reload(self.model, self.model_config)
        for module, hook in self.hooks:
            module.process_weights_after_loading = hook
        if hasattr(self.model, '_weights_finalized'):
            self.model._weights_finalized = False
        self.model.process_weights_after_loading()
        self.finished = True


def worker_extension():
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

    class DeepseekV41WorkerExtension(vLLMColocateWorkerExtension):
        def __new__(cls, **kwargs):
            install_reload_metadata_hook()
            return super().__new__(cls, **kwargs)

        def update_weights_from_ipc(
            self, peft_config=None, base_sync_done=False, use_shm=False
        ):
            from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
                BucketedWeightReceiver,
            )

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
            transport = BucketedWeightReceiver(
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
