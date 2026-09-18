# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Nested configuration field validation and explicit name projection."""
import json
from pathlib import Path


def read_config(path):
    path = Path(path)
    return json.loads((path / 'config.json' if path.is_dir() else path).read_text())


def nested_fields(release, model_type, sections):
    if release.get('model_type') != model_type:
        raise ValueError(f'Expected model_type={model_type}')
    for section in sections.split():
        if not isinstance(release.get(section), dict):
            raise ValueError(f'Missing nested {section}')
    return [release[section] for section in sections.split()]


def require_fields(fields, fixed, checks):
    for key, expected in fixed.items():
        if fields.get(key) != expected:
            raise ValueError(f'Unsupported {key}: expected {expected}')
    for invalid, message in checks:
        if invalid():
            raise ValueError(message)


def project_fields(fields, mapping):
    pairs = (
        item.split('=') if '=' in item else (item, item) for item in mapping.split()
    )
    return {target: fields[source] for target, source in pairs}
