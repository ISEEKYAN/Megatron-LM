# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native reload lifecycle boundaries for CPU receiver tests."""
import sys
from types import ModuleType
from types import SimpleNamespace as NS


def install_reload(monkeypatch, model, initialize=None):
    model._ds41_reload_metadata = True
    model.process_weights_after_loading = lambda: None
    infos = {}
    for layer in model.modules():
        info = NS(loaded_weights=[], kernel_tensors=None, can_load=lambda: False)
        info.reset = lambda info=info: info.loaded_weights.clear()
        infos[layer] = info
    name = 'vllm.model_executor.model_loader.reload'
    reload = ModuleType(name)
    layerwise = ModuleType(name + '.layerwise')
    layerwise.get_layerwise_info = infos.__getitem__
    layerwise.LOADING_LAYERS = set()

    def place(layer, info):
        for name, param in info.kernel_tensors[0].items():
            setattr(layer, name, param)

    layerwise._place_kernel_tensors = place
    reload.layerwise = layerwise
    reload.initialize_layerwise_reload = initialize or (lambda model: None)
    reload.finalize_layerwise_reload = lambda model, config: None
    parent = None
    for name in ('vllm', 'vllm.model_executor', 'vllm.model_executor.model_loader'):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
        if parent is not None:
            setattr(parent, name.rsplit('.', 1)[-1], package)
        parent = package
    parent.reload = reload
    monkeypatch.setitem(sys.modules, reload.__name__, reload)
    monkeypatch.setitem(sys.modules, layerwise.__name__, layerwise)
    return infos, reload
