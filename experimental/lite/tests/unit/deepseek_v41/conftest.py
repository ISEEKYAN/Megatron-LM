# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest


@pytest.fixture
def v41_core_te(transformer_engine_import_stub):
    # Resolve Core's optional TE imports before installing the Lite import stub.
    import megatron.core.fp8_utils  # noqa: F401
    import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
    import megatron.core.transformer.hyper_connection  # noqa: F401

    transformer_engine_import_stub()
