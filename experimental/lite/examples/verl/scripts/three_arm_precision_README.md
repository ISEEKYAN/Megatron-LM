# Three-arm Muon precision harness (real workload)

Verifies MLite's **dist_opt Muon** on a real Qwen3.5-35B-A3B verl-SFT workload, on
≤8 H100 (CW). Uses the ready-made verl SFT launcher (`run_qwen3moe_sft.sh`) — no
custom TP×PP×EP×CP harness — per the authoritative task guide.

## Arms (fixed same-contract: model / data order / init / seed / tokens / LR)

| ARM | optimizer_algorithm | muon_tp_mode | purpose |
|-----|---------------------|--------------|---------|
| `adamw` | adamw | (n/a) | A/B baseline |
| `muon` | muon | **distributed** | DistOpt Muon doing *true* cross-TP Newton-Schulz (TP2) — the primary arm |
| `muon_local` | muon | blockwise | per-shard local-NS contrast (see below) |

Primary verdict (bayan's criterion): **Muon must be no worse than AdamW** on the
loss trajectory. Evidence = per-step JSONL loss emitted by each GPU run, NOT any
config assertion.

## Key facts established before burning GPU (root-cause, `emerging_optimizers.py`)

1. The dist_opt Muon **implementation is NVIDIA Megatron-Core**
   (`megatron/core/optimizer/muon.py` → `emerging_optimizers.py`). The ISEEKYAN
   fork does **not** ship it, so `MEGATRON_ROOT` must point at an NVIDIA mcore
   checkout (default: `ds4-csacp-parity-eaa5b486d/mcore@fd1121b`). MLite's Muon
   forwards directly to `get_megatron_muon_optimizer`, so "MLite DistOpt Muon" **is**
   Megatron's TensorParallel Muon by construction (the arm-3 "alignment" is a
   construction-level identity, corroborated by matching short-run loss).
2. `muon_tp_mode ∈ {blockwise, duplicated, distributed}`. **blockwise is aliased to
   `duplicated`** internally (`partition_dim=None`) → per-shard *local* NS, i.e. not
   distributed. Only `distributed` runs the true cross-TP Newton-Schulz that the
   task requires (this is why the primary `muon` arm uses `distributed`, and
   `muon_local` keeps `blockwise` purely as a contrast).

## Gates (run on 0 GPU before any 8×H100 allocation)

`submit_three_arm_precision.sh` runs two gates and only then submits:

- **2a config dry-run** — resolves the `torchrun` command per arm, printing the
  distinct `optim.optimizer` / `muon_tp_mode`. Command-resolution only; *not* a
  precision claim.
- **2b init-chain gate (container, CPU)** — imports the real chain
  (`megatron.core`+`megatron.lite` namespace merge, the NVIDIA Muon getter,
  `verl_mlite` engine) and constructs `OptimizerConfig` for each arm, asserting the
  resolved `optimizer_algorithm`/`muon_tp_mode`. Catches env/namespace breakage on
  0 GPU. Still *not* a precision result.

The precision verdict itself only exists once the GPU arms have emitted loss.

## Run

```bash
# from cw-dfw-cs-001-login-02
GATE_ONLY=1 bash three_arm_precision/submit_three_arm_precision.sh   # gates only
ARMS="adamw muon" bash three_arm_precision/submit_three_arm_precision.sh   # A/B
```
