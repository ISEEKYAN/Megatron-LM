# Training mathematics specification verification

CPU-only verification; no model implementation, native backward, distributed
optimizer or GPU test was run. Interface and tolerance approval remains subject
to specification review.

Report bytes at `/tmp/deepseek-v41-report.pdf` matched the SHA256 in the contract.
Read report sections 2.1–2.5, 3.1.2–3.1.3 and optimizer constants in the pinned A4
contract. Differentiation cards are independently derived, not inference backward.

Run from repository root:

```sh
python experimental/lite/docs/specs/verify_deepseek_v41_training_cards.py
git diff --check
```

Both exited 0. The first printed:

```text
8 smooth cards: both finite-difference steps passed; STE declared VJP checked; Sinkhorn two-step and AdamW vectors passed
```

The STE check verifies only the declared algebraic card, not any custom autograd
implementation. The two-step Sinkhorn check uses floating scalar arithmetic;
A4's nonuniform high-precision reference remains mandatory downstream.

For policy verification, extract these four files from commit
`766bc22d1396c30a6c7d08deabe64f0a56e84d8f`, under
`experimental/lite/docs/plans/`, into one temporary directory:
`validate_deepseek_v41_plan.py`, `deepseek_v41_phases_bg_v4.md`,
`deepseek_v41_dependencies.json`, `deepseek_v41_decisions.json`.
Run `python <directory>/validate_deepseek_v41_plan.py --active-decision ID`
separately for every row below. The actual extraction directory was
`/tmp/b2s-plan`. Successful runs each rejected the validator's nine mutations.

| Decision IDs (each invoked separately) | Actual exit | Meaning |
|---|---:|---|
| O01, O02, O03, O04, O07, O10, O11, O13, O14, O16, O17 | 0 each | Resolved active policies; structural check only |
| O08 | 1 | Expected `active decision O08 remains OPEN` |
| O09 | 1 | Expected `active decision O09 remains OPEN` |
| O12 | 1 | Expected `active decision O12 remains OPEN` |

The pinned task matrix lists O16 as the direct specification blocker, and O12
as the subsequent training implementation blocker. O16 is resolved in A8's
decision record. The static `blocked_by_open` field was not recomputed there;
it does not reopen O16. O08/O09/O12 remain unknown and are not silently activated
by the mathematical cards. This distinction requires explicit reviewer scrutiny.
