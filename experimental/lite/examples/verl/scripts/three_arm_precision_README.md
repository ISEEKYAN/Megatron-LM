# Three-arm Muon precision harness (real workload)

Verifies MLite's **DistOpt Muon** vs **FSDP2 Muon** on a real Qwen3-30B-A3B verl-SFT
workload, on ≤8 H100 (CW). Uses the ready-made verl SFT launcher
(`run_qwen3moe_sft.sh`) — no custom TP×PP×EP×CP harness — per the authoritative task
guide.

## Model proxy: Qwen3-30B-A3B (not Qwen3.5)

bayan's redirect (2026-07-22) is to verify Muon three-arm parity on a **non-3.5
Qwen3** so the packed-THD path does **not** require the FLA causal-conv kernel that
blocks Qwen3.5-35B-A3B's GatedDeltaNet in this container.

- Qwen3-30B-A3B has `model_type=qwen3_moe`, **standard attention, no GatedDeltaNet /
  linear attention** → its THD path needs no FLA. mlite supports it directly
  (`qwen3_moe` package).
- mlite has **no dense qwen3 model** (the registry maps only `qwen3_moe`/`qwen2_moe`
  → `qwen3`; `model_type="qwen3"` dense is unregistered and the package is MoE-only).
  Qwen3-30B-A3B is therefore the faithful FLA-free "non-3.5 Qwen3" proxy.
- Muon three-arm parity is **model-agnostic**: it is about Newton-Schulz on
  TP-sharded 2D weights (attention QKV/O, expert gate/up/down), present in both
  dense and MoE, so the MoE proxy does not weaken the parity claim.

## Arms (fixed same-contract: model / data order / init / seed / tokens / LR)

| ARM | optimizer | backend | muon_tp_mode | purpose |
|-----|-----------|---------|--------------|---------|
| `adamw` | adamw | dist_opt | (n/a) | A/B baseline |
| `muon` | muon | **dist_opt** | **distributed** | DistOpt Muon = Megatron-Core TensorParallel Muon doing *true* cross-TP Newton-Schulz (TP2) |
| `muon_fsdp2` | muon | **fsdp2** | distributed | independent in-repo FSDP2 Muon lowering (full-gather → orthogonalize → reshard) |

Verdicts:
- **A/B (bayan's criterion): Muon must be no worse than AdamW** on the loss
  trajectory. Evidence = per-step JSONL loss from each GPU run.
- **Megatron vs DistOpt = bitwise by construction.** MLite's dist_opt Muon forwards
  directly to Megatron-Core `get_megatron_optimizer(optimizer="muon")`, so it *is*
  Megatron's TensorParallel Muon (same code path). No separate "Megatron-native"
  binary exists to diff — the identity is structural, corroborated by the loss.
- **FSDP2 = tolerance.** `muon_fsdp2` is a separate reimplementation
  (`primitive/optimizers/fsdp2/muon.py`), compared to the `muon` arm within numerical
  tolerance on the loss trajectory.

## Correctness fix carried by this harness: `muon_tp_mode` propagation

`build_dist_opt_optimizer_config` (`primitive/optimizers/megatron_wrap.py`) used to
build the Megatron-Core `OptimizerConfig` copying only lr/betas/offload — it **dropped
every `muon_*` field**, so mcore silently fell back to its default
`muon_tp_mode="blockwise"` (per-shard *local* NS), degrading a requested
`muon_tp_mode=distributed` back to local NS. The fix forwards all ten `muon_*` fields
when `optimizer_algorithm=="muon"`. The init-chain gate (2b below) asserts the
propagation directly by building the CoreOptimizerConfig on 0 GPU.

## Gates (run on 0 GPU before any 8×H100 allocation)

`submit_three_arm_precision.sh` runs two gates and only then submits:

- **2a config dry-run** — resolves the `torchrun` command per arm, printing the
  distinct `optim.optimizer` / `muon_tp_mode` / `impl_cfg.optimizer`. Command
  resolution only; *not* a precision claim.
- **2b init-chain gate (container, CPU)** — imports the real chain
  (`megatron.core`+`megatron.lite` namespace merge, the NVIDIA Muon getter,
  `verl_mlite` engine), constructs `OptimizerConfig` for each arm, **and builds the
  Megatron-Core `CoreOptimizerConfig` to assert `muon_tp_mode` propagates** (the fix).
  Catches env/namespace/propagation breakage on 0 GPU. Still *not* a precision result.

The precision verdict itself only exists once the GPU arms have emitted loss.

## External dependency: `emerging_optimizers` (dist_opt muon arm only)

Megatron-Core's dist_opt Muon calls into the external **`emerging_optimizers`** package
(the Newton-Schulz kernels). The container does **not** ship it, and neither the
`adamw` arm nor the self-contained `muon_fsdp2` arm needs it. Provide it to the
`muon` arm via `EMERGING_OPT_SITE` (a `pip install --target=<dir> --no-deps
emerging-optimizers==0.3.0` site), which the node script prepends to `PYTHONPATH`.

## Run

```bash
# from cw-dfw-cs-001-login-02
GATE_ONLY=1 bash submit_three_arm_precision.sh                          # gates only
ARMS="adamw muon muon_fsdp2" bash submit_three_arm_precision.sh         # all three
```
