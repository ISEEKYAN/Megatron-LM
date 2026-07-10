# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Create deterministic GSM8K-schema arithmetic data for bounded RL smokes."""

from __future__ import annotations

import argparse
from pathlib import Path


_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def build_rows(*, size: int, split: str, index_offset: int) -> list[dict]:
    """Build rule-reward rows without requiring network dataset access."""
    if size <= 0:
        raise ValueError("size must be positive")
    rows = []
    for local_index in range(size):
        index = index_offset + local_index
        crates = 17 + (index % 19)
        items_per_crate = 23 + ((index * 7) % 29)
        damaged = 31 + ((index * 11) % 47)
        bundles = 3 + (index % 7)
        total = crates * items_per_crate - damaged
        answer = total // bundles
        remainder = total % bundles
        question = (
            f"A warehouse has {crates} crates with {items_per_crate} parts in each crate. "
            f"It discards {damaged} damaged parts, then makes bundles of {bundles} parts. "
            f"How many complete bundles can it make, and how many parts remain? "
            f"Report the value complete_bundles * 100 + remaining_parts. {_INSTRUCTION}"
        )
        ground_truth = str(answer * 100 + remainder)
        rows.append(
            {
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": split,
                    "index": index,
                    "answer": f"#### {ground_truth}",
                    "question": question,
                    "provenance": "deterministic-arithmetic-smoke",
                },
            }
        )
    return rows


def write_dataset(output_dir: Path, *, train_size: int, test_size: int) -> None:
    from datasets import Dataset

    output_dir.mkdir(parents=True, exist_ok=False)
    train = Dataset.from_list(build_rows(size=train_size, split="train", index_offset=0))
    test = Dataset.from_list(
        build_rows(size=test_size, split="test", index_offset=train_size)
    )
    train.to_parquet(output_dir / "train.parquet")
    test.to_parquet(output_dir / "test.parquet")
    print(
        f"DS4_MATH_SMOKE_DATA_READY train={train.num_rows} test={test.num_rows} "
        f"output={output_dir}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=16)
    args = parser.parse_args()
    write_dataset(args.output_dir, train_size=args.train_size, test_size=args.test_size)


if __name__ == "__main__":
    main()
