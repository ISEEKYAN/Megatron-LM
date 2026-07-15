# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deterministic *variable-length* messages-format SFT data for THD+PP validation.

The THD pipeline P2P shape bug only manifests when consecutive micro-batches
carry *different* packed token counts: a fixed recv buffer sized from the first
micro-batch mismatches later ones (NCCL irecv requires exact element counts).
Uniform sequence lengths hide it. This generator therefore spreads per-sample
lengths across a wide range so that dynamic-bsz THD packing produces
micro-batches with clearly unequal total-token counts.

Schema: one `messages` column of ``[{"role", "content"}, ...]`` — the VERL
messages format consumed by ``verl.trainer.sft_trainer``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


# A pool of filler sentences; repeating a variable count grows a sample's token
# length deterministically without any network/tokenizer dependency.
_FILLER = (
    "The warehouse crew logged every crate before the night shift began. "
    "Each pallet was scanned twice to keep the running inventory honest. "
    "A short delay upstream rippled into the packing lane by mid-morning. "
    "Operators rotated between stations so no single bottleneck formed. "
)


def _content(*, base_words: int, reps: int) -> str:
    """Build a message body whose length scales with ``reps``."""
    body = (_FILLER * max(reps, 1)).strip()
    # Trim/extend to a rough word budget so lengths spread smoothly.
    words = body.split()
    if base_words < len(words):
        words = words[:base_words]
    return " ".join(words)


def build_rows(*, size: int, split: str, index_offset: int) -> list[dict]:
    if size <= 0:
        raise ValueError("size must be positive")
    rows = []
    for local_index in range(size):
        index = index_offset + local_index
        # Wide, deterministic spread of lengths: short (~30w) .. long (~1500w).
        # The non-linear step makes adjacent samples differ, so however the
        # packer groups them the resulting micro-batch token totals vary.
        reps = 1 + (index % 23)
        base_words = 30 + ((index * 37) % 1500)
        user_body = _content(base_words=base_words, reps=reps)
        # Assistant target length varies independently of the prompt length.
        asst_reps = 1 + ((index * 5) % 17)
        asst_words = 20 + ((index * 53) % 900)
        asst_body = _content(base_words=asst_words, reps=asst_reps)
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"[sample {index}] Summarize: {user_body}",
                    },
                    {
                        "role": "assistant",
                        "content": f"Summary {index}: {asst_body}",
                    },
                ],
                "extra_info": {
                    "split": split,
                    "index": index,
                    "approx_user_words": base_words,
                    "approx_asst_words": asst_words,
                    "provenance": "deterministic-varlen-sft-thd",
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
    # Report the realized length spread so the operator can confirm the data is
    # genuinely variable-length before firing the GPU A/B.
    approx = sorted(r["extra_info"]["approx_user_words"] for r in train)
    print(
        f"VARLEN_SFT_DATA_READY train={train.num_rows} test={test.num_rows} "
        f"user_words[min={approx[0]},med={approx[len(approx) // 2]},max={approx[-1]}] "
        f"output={output_dir}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=16)
    args = parser.parse_args()
    write_dataset(args.output_dir, train_size=args.train_size, test_size=args.test_size)


if __name__ == "__main__":
    main()
