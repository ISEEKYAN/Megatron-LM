# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Keep explicit prefix-cache disables through the actual VERL CLI helper."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.optional
COMPAT = Path(__file__).resolve().parents[3] / 'examples/verl/verl_mlite/compat.py'


@pytest.mark.parametrize('key', ['enable_prefix_caching', 'enable-prefix-caching'])
@pytest.mark.parametrize('value', [False, True, None])
def test_prefix_cache_cli_preserves_explicit_disable(monkeypatch, key, value):
    """The server's pre-bound helper must emit a negative flag for False."""
    utils = pytest.importorskip('verl.workers.rollout.vllm_rollout.utils')
    spec = importlib.util.spec_from_file_location('cli_compat_test', COMPAT)
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    monkeypatch.delenv('VERL_MLITE_VLLM_SITE', raising=False)
    monkeypatch.setenv('VERL_MLITE_HF_CONFIG_MODEL_TYPE', 'deepseek_v41')
    original = utils.build_cli_args_from_config
    consumer = SimpleNamespace(build_cli_args_from_config=original)
    monkeypatch.setitem(sys.modules, compat._VLLM_ASYNC_SERVER_MODULE, consumer)
    # Restore the real dependency after this test's in-place compatibility patch.
    monkeypatch.setattr(utils, 'build_cli_args_from_config', original)
    config = {key: value, 'unrelated_store_true': False, 'max_num_seqs': 32}
    expected = original({k: v for k, v in config.items() if k != key})
    if value is False:
        expected.append('--no-enable-prefix-caching')
    else:
        expected = original(config)
    before = dict(config)
    assert compat._patch_verl_vllm_prefix_caching_cli()
    assert consumer.build_cli_args_from_config(config) == expected
    assert config == before
    assert not compat._patch_verl_vllm_prefix_caching_cli()
