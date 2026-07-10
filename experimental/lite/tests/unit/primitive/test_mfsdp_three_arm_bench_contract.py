# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
BENCH = ROOT / "experimental/lite/tests/smoke/primitive/test_mfsdp_three_arm_bench.py"
RUNNER = ROOT / "experimental/lite/tests/run_mfsdp_three_arm_bench.sh"


def _assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    return values


def test_three_arm_bench_freezes_comparable_protocol():
    values = _assignments(BENCH)

    assert values["_MCORE_COMMIT"] == "00309a0199dc590060aa0995b6f4a371d8db9761"
    assert values["_MLITE_BASE_COMMIT"] == "62295f9b306d70a8180e907b7c51b3ef293ea007"
    assert values["_BENCHMARK_PROTOCOL_COMMIT"] == (
        "5338da72e102214745d4feacc445a152c512c30a"
    )
    assert values["_ARMS"] == ("mcore_mfsdp", "mlite_mfsdp", "mlite_fsdp2")
    assert values["_TOPOLOGY"] == (2, 2, 1, 2, 2)
    assert values["_WARMUP_STEPS"] >= 5
    assert values["_MEASURE_STEPS"] >= 20
    assert values["_PRECISION_STEPS"] >= 3
    assert values["_OPTIMIZER"] == "torch.optim.AdamW"
    assert values["_COMPUTE_DTYPE"] == "bfloat16"
    assert values["_MAIN_PARAM_DTYPE"] == "bfloat16"


def test_mlite_ablation_matrix_names_every_required_feature():
    values = _assignments(BENCH)
    ablations = values["_MLITE_ABLATIONS"]

    assert tuple(ablations) == (
        "bucket",
        "ag_overlap",
        "rs_overlap",
        "prefetch",
        "double_buffer",
        "ub_zero_copy",
        "nccl_registered_buffer",
        "gdr",
    )
    assert all(tuple(ablations[name]) == (False, True) for name in ablations)


def test_runner_is_slurm_only_and_exposes_both_signoff_modes():
    source = RUNNER.read_text()

    assert 'SLURM_JOB_ID' in source
    assert 'three-arm' in source
    assert 'ablation' in source
    assert 'NPROC_PER_NODE:-8' in source


def test_mcore_arm_preserves_named_tp_dimension_in_both_meshes():
    source = BENCH.read_text()

    assert 'mesh_dim_names=("dp_cp", "tp")' in source
    assert 'tp_dim="tp"' in source
    assert "_dense_dp_tp_rank_mesh(ps)" in source
    assert "_expert_dp_tp_rank_mesh(ps)" in source
    assert "use_local_synchronization=True" in source


def test_arm_construction_is_synchronized_before_next_global_group_sequence():
    source = BENCH.read_text()

    assert source.count("_synchronize_arm_build(") >= 3
