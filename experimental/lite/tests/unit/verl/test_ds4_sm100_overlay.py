from pathlib import Path

import pytest


_EXAMPLES = Path(__file__).parents[3] / "examples/verl"
_SLURM = _EXAMPLES / "slurm"
_PATCHES = _EXAMPLES / "patches"


def test_sm100_dependency_contract_requires_blackwell_and_expected_exports() -> None:
    from examples.verl.ds4_sm100_env import validate_dependency_contract

    report = validate_dependency_contract(
        capability=(10, 0),
        cudnn_frontend_version="1.26.0",
        flash_mla_sparse_fwd=object(),
        indexer_fwd_sm100=object(),
    )

    assert report == {
        "capability": "10.0",
        "cudnn_frontend": "1.26.0",
        "flash_mla_sparse_fwd": True,
        "indexer_fwd_sm100": True,
    }


@pytest.mark.parametrize(
    ("capability", "version", "flash", "indexer", "message"),
    [
        ((9, 0), "1.26.0", object(), object(), "SM100"),
        ((10, 0), "1.25.0", object(), object(), "1.26.0"),
        ((10, 0), "1.26.0", None, object(), "flash_mla_sparse_fwd"),
        ((10, 0), "1.26.0", object(), None, "indexer_fwd_sm100"),
    ],
)
def test_sm100_dependency_contract_fails_closed(
    capability: tuple[int, int],
    version: str,
    flash: object | None,
    indexer: object | None,
    message: str,
) -> None:
    from examples.verl.ds4_sm100_env import validate_dependency_contract

    with pytest.raises(RuntimeError, match=message):
        validate_dependency_contract(
            capability=capability,
            cudnn_frontend_version=version,
            flash_mla_sparse_fwd=flash,
            indexer_fwd_sm100=indexer,
        )


def test_sm100_overlay_build_is_pinned_and_does_not_mutate_rollout_overlay() -> None:
    versions = (_SLURM / "ds4_sm100_versions.env").read_text()
    build = (_SLURM / "build_ds4_sm100_overlay.sbatch").read_text()

    assert "NGC_IMAGE=docker://nvcr.io#nvidia/pytorch:26.06-py3" in versions
    assert "VLLM_COMMIT=cd0de48d0883ecb8e1ef350a99baa0c158f58e82" in versions
    assert "FLASHMLA_REPO=https://github.com/vllm-project/FlashMLA.git" in versions
    assert "FLASHMLA_COMMIT=a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b" in versions
    assert "MLITE_OVERLAY=" in versions
    assert "VLLM_OVERLAY=" in versions
    assert "FLASH_MLA_DISABLE_SM90=1" in build
    assert "CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl" in build
    assert "flashmla-a6ec-standalone-import.patch" in build
    assert 'git -C "${FLASHMLA_SRC}" apply --check' in build
    assert "--force-reinstall" in build
    assert (
        "python -m pip install --force-reinstall --no-build-isolation --no-deps "
        '"${flashmla_src}"' in build
    )
    assert "gb200_vllm_overlay.pth" in build
    assert "nvidia_cutlass_dsl/python_packages" in build
    assert "python -m examples.verl.ds4_sm100_env probe" in build
    assert 'source "${VLLM_OVERLAY}/bin/activate"' not in build
    assert "cp -r" not in build
    assert ".whl" not in build


def test_flashmla_patch_only_removes_the_broken_optional_export() -> None:
    patch = (_PATCHES / "flashmla-a6ec-standalone-import.patch").read_text()

    assert "-    flash_mla_with_kvcache_fp8," in patch
    assert "flash_mla_sparse_fwd" not in patch
    assert patch.count("diff --git") == 1


def test_formal_mlite_forward_launcher_probes_then_runs_same_prompt_driver() -> None:
    script = (_SLURM / "run_ds4_mlite_bf16_forward.sbatch").read_text()

    assert "#SBATCH --gpus-per-node=4" in script
    assert "python -m examples.verl.ds4_sm100_env probe" in script
    assert "python -m torch.distributed.run --standalone --nproc_per_node=4" in script
    assert "-m examples.verl.ds4_resync_tp4 collect-mlite" in script
    assert "--model '${CHECKPOINT_DIR}'" in script
    assert "--output '${OUTPUT_DIR}/mlite.pt'" in script
    assert "--fp8-output '${OUTPUT_DIR}/resync-checkpoint'" in script
    assert "python -m examples.verl.ds4_sm100_env verify-payload" in script
    assert "--expected-prompts 36" in script
    assert "DS4_MLITE_SM100_FORWARD_OK" in script
    assert "COMPARE_ONLY" not in script
