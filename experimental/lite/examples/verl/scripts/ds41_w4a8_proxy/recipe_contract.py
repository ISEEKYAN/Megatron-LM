# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Validate prepared precision metadata and explicit rollout graph settings."""


def rollout_settings(config, environ):
    precision = environ.get('DS41_PRECISION', 'release')
    if precision not in ('release', 'official'):
        raise ValueError('DS41_PRECISION must be release or official')
    text = config.get('text_config', config)
    overrides = {'head_dtype': 'float32'}
    if precision == 'official':
        if (text.get('expert_activation_group_size'), text.get('wo_a_dtype')) != (
            32,
            'bf16',
        ):
            raise ValueError('Official recipe requires prepared g32/BF16 wo_a metadata')
        overrides.update(expert_activation_group_size=32, wo_a_dtype='bf16')
    graph = environ.get('DS41_CUDAGRAPH', 'none')
    if graph not in ('none', 'full_and_piecewise'):
        raise ValueError('DS41_CUDAGRAPH must be none or full_and_piecewise')
    compilation = {'mode': 0, 'cudagraph_mode': 'NONE'}
    if graph != 'none':
        capture = int(environ.get('DS41_CUDAGRAPH_MAX_CAPTURE', '16'))
        if capture != 16:
            raise ValueError('DS41 graph recipe requires capture upper bound 16')
        compilation.update(
            cudagraph_mode='FULL_AND_PIECEWISE', max_cudagraph_capture_size=capture
        )
    return dict(
        enforce_eager=graph == 'none',
        compilation_config=compilation,
        hf_overrides=overrides,
        breakable=graph != 'none',
        precision=precision,
    )


def hydra_mapping(mapping):
    return '{' + ','.join(f'{key}:{value}' for key, value in mapping.items()) + '}'
