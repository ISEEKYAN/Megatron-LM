#!/usr/bin/env python3
"""Reject defensive attribute fallbacks in MLite-owned model implementations."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "experimental/lite"
DEFAULT_ALLOWLIST = Path(__file__).with_name("model_structure_fallback_allowlist.json")
MODEL_PACKAGE_PREFIX = "experimental/lite/megatron/lite/model/"
TESTS_PREFIX = "experimental/lite/tests/"
OWNED_STRUCTURE_ATTRIBUTES = {
    "_local_tpe_list",
    "cross_entropy_fusion",
    "deterministic",
    "dispatcher",
    "embed",
    "head",
    "layer_indices",
    "layers",
    "model",
    "mtp",
    "mtp_embed",
    "norm",
    "recompute_modules",
    "sp_params",
}


@dataclass(frozen=True)
class Callsite:
    signature: str
    path: str
    line: int
    column: int


class _FallbackVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.scope: list[str] = []
        self.callsites: list[Callsite] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node)

    def _visit_scope(
        self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        kind = None
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 3
        ):
            kind = "getattr-default"
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id == "hasattr"
            and len(node.args) >= 2
        ):
            kind = "hasattr"
        if kind is not None:
            receiver = ast.unparse(node.args[0])
            attribute_node = node.args[1]
            if isinstance(attribute_node, ast.Constant) and isinstance(
                attribute_node.value, str
            ):
                attribute = attribute_node.value
            else:
                attribute = f"<dynamic:{ast.unparse(attribute_node)}>"
            if not (
                self.path.startswith(MODEL_PACKAGE_PREFIX)
                or attribute in OWNED_STRUCTURE_ATTRIBUTES
            ):
                self.generic_visit(node)
                return
            qualname = ".".join(self.scope) or "<module>"
            signature = f"{self.path}:{qualname}:{kind}:{receiver}:{attribute}"
            self.callsites.append(
                Callsite(
                    signature=signature,
                    path=self.path,
                    line=node.lineno,
                    column=node.col_offset + 1,
                )
            )
        self.generic_visit(node)


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def scan(source_root: Path, repo_root: Path) -> list[Callsite]:
    callsites: list[Callsite] = []
    for path in sorted(source_root.rglob("*.py")):
        relative = _relative_path(path, repo_root)
        if relative.startswith(TESTS_PREFIX):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except SyntaxError as exc:
            raise ValueError(f"cannot parse {relative}: {exc}") from exc
        visitor = _FallbackVisitor(relative)
        visitor.visit(tree)
        callsites.extend(visitor.callsites)
    return callsites


def load_allowlist(path: Path) -> dict[str, tuple[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain an 'entries' list")
    allowed: dict[str, tuple[int, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{path} entries must be objects")
        signature = entry.get("signature")
        reason = entry.get("reason")
        count = entry.get("count", 1)
        if not isinstance(signature, str) or not signature:
            raise ValueError(f"{path} entry has no signature")
        if signature in allowed:
            raise ValueError(f"{path} contains duplicate signature: {signature}")
        if not isinstance(reason, str) or len(reason.strip()) < 12:
            raise ValueError(
                f"{path} entry needs a specific boundary reason: {signature}"
            )
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"{path} entry has invalid count: {signature}")
        allowed[signature] = (count, reason)
    return allowed


def check(
    callsites: Iterable[Callsite], allowed: dict[str, tuple[int, str]]
) -> list[str]:
    callsites = list(callsites)
    actual = Counter(site.signature for site in callsites)
    errors: list[str] = []
    for site in callsites:
        allowance = allowed.get(site.signature)
        if allowance is None:
            errors.append(
                f"{site.path}:{site.line}:{site.column}: MLITE001 {site.signature} is not "
                "an approved interface boundary"
            )
    for signature, (expected_count, _reason) in sorted(allowed.items()):
        actual_count = actual.get(signature, 0)
        if actual_count == 0:
            errors.append(f"MLITE002 stale allowlist entry: {signature}")
        elif actual_count != expected_count:
            errors.append(
                f"MLITE003 allowlist count mismatch for {signature}: "
                f"expected {expected_count}, found {actual_count}"
            )
    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        callsites = scan(args.source_root.resolve(), args.repo_root.resolve())
        allowed = load_allowlist(args.allowlist.resolve())
        errors = check(callsites, allowed)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"MLite model-structure fallback check failed: {exc}")
        return 2
    if errors:
        print("MLite model-structure fallback check failed:")
        for error in errors:
            print(f"  {error}")
        print(
            "Use direct attribute access for MLite-owned structure. Register only genuine "
            "interface boundaries in the explicit allowlist."
        )
        return 1
    print(
        f"MLite model-structure fallback check passed: {len(callsites)} approved boundary "
        f"call(s), {len(allowed)} allowlist signature(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
