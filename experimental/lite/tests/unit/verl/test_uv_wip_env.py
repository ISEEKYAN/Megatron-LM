from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[5]
ENV_SH = REPO_ROOT / "experimental/lite/examples/verl/env/uv-wip/env.sh"
BOOTSTRAP_SH = ENV_SH.with_name("bootstrap.sh")
EXCLUDES = ENV_SH.with_name("uv-excludes.txt")
SOURCE_REQUIREMENTS = ENV_SH.with_name("source-requirements.txt")


def _source_roots(tmp_path: Path) -> dict[str, str]:
    site = tmp_path / "site"
    verl = tmp_path / "verl-src"
    megatron = tmp_path / "megatron-src"
    mlite = tmp_path / "mlite-src"
    (site / "vllm").mkdir(parents=True)
    (site / ".uv-wip-ready").touch()
    (verl / "verl").mkdir(parents=True)
    (megatron / "megatron/core").mkdir(parents=True)
    (mlite / "experimental/lite/megatron/lite").mkdir(parents=True)
    (mlite / "experimental/lite/examples/verl/verl_mlite").mkdir(parents=True)
    return {
        "UV_WIP_SITE": str(site),
        "VERL_ROOT": str(verl),
        "MEGATRON_ROOT": str(megatron),
        "MLITE_ROOT": str(mlite),
    }


def test_env_sh_exports_online_source_order_and_execs_command(tmp_path: Path) -> None:
    roots = _source_roots(tmp_path)
    original_pythonpath = tmp_path / "existing"
    original_pythonpath.mkdir()
    command = (
        "import json, os; "
        "print(json.dumps({k: os.environ[k] for k in "
        "['PYTHONPATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CC', 'CXX', 'PATH']}))"
    )

    result = subprocess.run(
        ["bash", str(ENV_SH), sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            **roots,
            "PYTHONPATH": str(original_pythonpath),
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
        },
    )

    exported = json.loads(result.stdout)
    mlite = Path(roots["MLITE_ROOT"])
    assert exported["PYTHONPATH"].split(os.pathsep) == [
        roots["UV_WIP_SITE"],
        roots["VERL_ROOT"],
        roots["MEGATRON_ROOT"],
        str(mlite / "experimental/lite"),
        str(mlite / "experimental/lite/examples/verl"),
        str(original_pythonpath),
    ]
    assert exported["OMP_NUM_THREADS"] == "1"
    assert exported["MKL_NUM_THREADS"] == "1"
    assert exported["CC"] == "/usr/bin/gcc"
    assert exported["CXX"] == "/usr/bin/g++"
    assert exported["PATH"].split(os.pathsep)[:2] == ["/usr/local/cuda/bin", "/usr/bin"]


def test_env_sh_fails_before_exec_when_source_contract_is_missing(
    tmp_path: Path,
) -> None:
    roots = _source_roots(tmp_path)
    del roots["VERL_ROOT"]

    result = subprocess.run(
        ["bash", str(ENV_SH), sys.executable, "-c", "print('must not run')"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **roots},
    )

    assert result.returncode == 64
    assert "VERL_ROOT" in result.stderr
    assert "must not run" not in result.stdout


def test_env_sh_rejects_an_unpublished_uv_site(tmp_path: Path) -> None:
    roots = _source_roots(tmp_path)
    (Path(roots["UV_WIP_SITE"]) / ".uv-wip-ready").unlink()

    result = subprocess.run(
        ["bash", str(ENV_SH), sys.executable, "-c", "print('must not run')"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **roots},
    )

    assert result.returncode == 64
    assert ".uv-wip-ready" in result.stderr
    assert "must not run" not in result.stdout


def test_env_sh_expands_an_optional_torch_free_kernel_site(tmp_path: Path) -> None:
    roots = _source_roots(tmp_path)
    kernel_site = tmp_path / "kernel-site"
    cutlass_python = kernel_site / "nvidia_cutlass_dsl/python_packages"
    cutlass_python.mkdir(parents=True)

    result = subprocess.run(
        [
            "bash",
            str(ENV_SH),
            sys.executable,
            "-c",
            "import os; print(os.environ['PYTHONPATH'])",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **roots, "UV_WIP_KERNEL_SITE": str(kernel_site), "PYTHONPATH": ""},
    )

    entries = result.stdout.strip().split(os.pathsep)
    assert entries[-2:] == [str(cutlass_python), str(kernel_site)]


def test_bootstrap_excludes_the_container_owned_runtime_stack() -> None:
    excluded = {
        line.strip()
        for line in EXCLUDES.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert {
        "torch",
        "torchvision",
        "torchaudio",
        "torchcodec",
        "torch-c-dlpack-ext",
        "functorch",
        "triton",
        "nvidia-cudnn-frontend",
        "nvidia-cutlass-dsl",
    } <= excluded


def test_bootstrap_resolves_then_atomically_publishes_an_audited_site() -> None:
    source = BOOTSTRAP_SH.read_text()

    assert "--python /usr/bin/python3" in source
    first_install = source.index("uv pip install")
    second_install = source.index("uv pip install", first_install + 1)
    assert first_install < source.index("--dry-run") < second_install
    assert 'PLAN_TARGET="$(mktemp -d' in source
    assert '--target "${PLAN_TARGET}"' in source
    assert '--requirements "${SOURCE_REQUIREMENTS}"' in source
    assert 'plan_status="$?"' in source
    assert 'resolution failed with status ${plan_status}' in source
    assert 'BUILD_TARGET="$(mktemp -d' in source
    assert '--target "${BUILD_TARGET}"' in source
    assert 'UV_WIP_SITE="${BUILD_TARGET}"' in source
    assert 'ln -s vllm/third_party/triton_kernels' in source
    assert "importlib.metadata.distributions(path=[site])" in source
    assert '"torch-c-dlpack-ext"' in source
    assert "name.startswith(\"nvidia-\")" in source
    ready_marker = source.index('touch "${BUILD_TARGET}/.uv-wip-ready"')
    publish = source.index('mv -T -- "${BUILD_TARGET}" "${UV_WIP_SITE}"')
    assert source.index("importlib.metadata.distributions(path=[site])") < ready_marker < publish


def test_online_verl_dependencies_are_resolved_without_installing_verl() -> None:
    requirements = {
        line.strip()
        for line in SOURCE_REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert "tensordict>=0.8.0,<=0.10.0,!=0.9.0" in requirements
    assert "ray[default]>=2.41.0" in requirements
    assert "numpy>=2" in requirements
    assert not any(requirement.startswith("numpy<") for requirement in requirements)
    assert not any(requirement.startswith(("verl", "megatron", "mlite")) for requirement in requirements)
