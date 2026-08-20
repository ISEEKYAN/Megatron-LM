from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


PATH = Path(__file__).parents[3] / "examples/verl/scripts/validate_deepseek_v4_dapo.py"
SPEC = spec_from_file_location("ds4_dapo_validator", PATH)
VALIDATOR = module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


@pytest.mark.parametrize(
    ("name", "torch_version", "cuda"),
    [
        ("nv26.05-cuda13.2", "2.12.0a0+5aff3928d8.nv26.05", "13.2"),
        ("ds4-vllm-align-cuda13.0", "2.13.0+cu130", "13.0"),
    ],
)
def test_release_profiles_are_explicit(name, torch_version, cuda):
    dependencies = {
        package: version
        for package, version in VALIDATOR.SUPPORTED_PROFILES[name].items()
        if package not in {"torch", "cuda"}
    }
    assert VALIDATOR.match_profile(dependencies, torch_version, cuda) == name
    assert VALIDATOR.match_profile(dependencies, torch_version, "12.8") is None
