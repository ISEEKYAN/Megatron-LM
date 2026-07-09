import pytest


def test_resync_format_accepts_only_explicit_contract_values() -> None:
    from megatron.lite.runtime.contracts.weights import ResyncFormat

    assert ResyncFormat.parse("bf16") is ResyncFormat.BF16
    assert ResyncFormat.parse("vllm_checkpoint") is ResyncFormat.VLLM_CHECKPOINT
    with pytest.raises(ValueError, match="resync_format"):
        ResyncFormat.parse("fp8")
