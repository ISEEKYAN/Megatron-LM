# Three-arm speed+memory summary (AC#4)

**Model**: Qwen3-30B-A3B (verl SFT, gsm8k parquet)  
**Topology**: 8 GPU, TP=2, EP=8, PP=1, CP=1, seed=1234, 20 steps  
**RUN_ROOT**: `runtime/task-1-13-5-5-4-speed-memory`  
**Reference**: Megatron-LM@4ce841ccd (mlite staged), mcore@fd1121b8f, verl.vllm023.sqsh

## Slurm jobs (all rc=0)

| Arm | Job ID | Optimizer | Backend | Key config |
| --- | ---: | --- | --- | --- |
| adamw | 14276873 | AdamW | dist_opt | `optimizer_cpu_offload=True`, `use_distributed_optimizer=True` |
| muon_distopt | 14277724 | Muon | dist_opt | `use_layer_wise_distributed_optimizer=True`, `optimizer_cpu_offload=False` |
| muon_fsdp2 | 14276875 | Muon | fsdp2 | FSDP2 Muon lowering |

## Results table

| Arm | Steps | Loss (step1→last) | Mean loss (steady) | Peak alloc GB | Peak reserved GB | Mean MFU (steady) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| adamw | 20 | 1.5100→0.3248 | 0.5145 | 31.34 | 32.80 | 0.001851 |
| muon_distopt | 20 | 1.5092→0.3422 | 0.5839 | 59.58 | 59.77 | 0.014193 |
| muon_fsdp2 | 20 | 1.5118→0.3222 | 0.5146 | 30.17 | 36.17 | 0.000845 |

JSONL paths:
- `adamw/q3moe_speed_mem_adamw.jsonl`
- `muon/q3moe_speed_mem_muon.jsonl`
- `muon_fsdp2/q3moe_speed_mem_muon_fsdp2.jsonl`

## Acceptance analysis

### (a) DistOpt muon vs FSDP2 muon precision

- Final-step loss: **0.3422 vs 0.3222** (Δ=0.020, ~6% relative).
- Step-1 loss aligned (~1.509). Curves diverge mildly mid-training (steps 3–8) then reconverge; FSDP2 tracks AdamW almost exactly.
- Steady-state mean loss gap: 0.0693 (inflated by mid-run bump on DistOpt path).
- **Verdict**: end-state loss close; full-trajectory not bitwise-identical but within same ballpark for 20-step proxy.

### (b) Muon peak memory vs AdamW

| Comparison | Result |
| --- | --- |
| FSDP2 muon (30.17 GB) vs AdamW (31.34 GB) | **Muon lower** (−1.17 GB, −3.7%) |
| DistOpt muon (59.58 GB) vs AdamW (31.34 GB) | **Muon higher** (+28.24 GB) |

**Fairness notes** (documented, not hidden):
- AdamW arm uses **CPU optimizer offload** (`optimizer_cpu_offload=True`, fraction=1.0) which suppresses GPU peak.
- DistOpt Muon **cannot** use generic CPU offload (fail-loud); runs no-offload with `use_layer_wise_distributed_optimizer=True`.
- Aligns with bayan note: verl 30B NS scratch already showed sharded Muon can still exceed Adam GPU peak.

### Speed (MFU proxy)

DistOpt Muon steady MFU ~7.7× AdamW/FSDP2 in this 20-step window (0.014 vs ~0.0018). Treat as indicative only (short run, warmup skew).

## Fixes applied during experiment

1. `mlite_engine._normalize_optimizer_name`: read VERL `optim.optimizer` (job 14276874 had silently fallen back to Adam).
2. DistOpt Muon: disable CPU offload in node script (match AC#3 nooffload).
3. `state.py`: add `intra_dist_opt_group` init (job 14277570 AttributeError).

## Reproduce

```bash
cd $RUN_ROOT/mlite-572b8a742/experimental/lite/examples/verl/scripts
export MLITE_REPO RUN_ROOT BASE
ARMS="adamw muon muon_fsdp2" bash submit_three_arm_speed_memory.sh
python summarize_three_arm_speed_memory.py --run-root $RUN_ROOT --output three_arm_speed_memory_RESULTS.md
```
