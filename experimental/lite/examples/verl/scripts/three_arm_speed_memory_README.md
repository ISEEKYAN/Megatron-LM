# Three-arm speed+memory harness (AC#4)

Qwen3-30B-A3B (`qwen3_moe`, FLA-free proxy) verl SFT on 8×H100. Three independent
torchrun arms with identical seed/batch/hyperparams; only optimizer differs.

| Arm | Optimizer | Backend |
| --- | --- | --- |
| `adamw` | AdamW | dist_opt |
| `muon` | Muon (layer-wise distributed optimizer ON) | dist_opt |
| `muon_fsdp2` | Muon | fsdp2 |

Environment (bayan 2026-07-23): `pytorch_26.04-py3.sqsh` +
`mlite-2604-verl-dsa-sm90-overlay`; Megatron-Core + `emerging_optimizers` from
snapshot `muon-mbridge-386bf7af6-r2` (`megatron-d64` / `emerging-b309`).

## Run (CW login node)

```bash
cd experimental/lite/examples/verl/scripts
GATE_ONLY=1 bash submit_three_arm_speed_memory.sh
ARMS="adamw muon muon_fsdp2" bash submit_three_arm_speed_memory.sh
```

Summarize after all arms complete:

```bash
python summarize_three_arm_speed_memory.py \
  --run-root /lustre/.../runtime/task-1-13-5-5-4-speed-memory \
  --output three_arm_speed_memory_RESULTS.md
```

The summarizer returns exit status 1 unless both Muon arms report lower peak
allocated memory than AdamW. It reports the DistOpt/FSDP2 loss gap for review;
this experiment has no separately specified numerical loss-gap threshold.
