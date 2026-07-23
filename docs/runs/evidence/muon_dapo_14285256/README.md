# Qwen3-30B-A3B DAPO 30-step reward A/B

Slurm job `14285256` ran both arms sequentially on 16 H100 GPUs and completed
with top-level state `COMPLETED`, exit code `0:0`, and no skipped arm. The
matched configuration used Qwen3-30B-A3B, seed 42, DAPO rewards, train batch
size 16, four responses per prompt, LR `1e-5`, constant scheduling, BF16 model
parameters, and FP32 optimizer state. The optimizer (`muon` or `adam`) was the
experimental variable.

The result **does not pass the Muon reward hard gate**:

| Statistic (10-step windows) | Muon | AdamW |
| --- | ---: | ---: |
| First-window mean | -0.962500 | -0.965625 |
| Last-window mean | -1.000000 | -0.246875 |
| Window gain | -0.037500 | +0.718750 |
| Linear slope per step | -0.001641 | +0.037236 |

Muon briefly reached `-0.875` in the first ten steps, then every reward from
step 11 through step 30 was `-1.0`. AdamW improved strongly after step 12,
reached positive batch rewards at steps 19 and 28, and retained a much higher
last-window mean. Therefore Muon neither showed sustained reward growth nor
matched AdamW over this 30-step run.

`muon_rewards.jsonl` and `adam_rewards.jsonl` are lossless normalized extracts
of all `critic/rewards/mean` points from the corresponding console logs.
`reward_curves.json` contains the window/slope statistics and explicit failed
verdict, and `reward_curves.svg` is the dual curve. `sacct.txt` records the
Slurm state. The worker step shown as cancelled is the expected Ray worker
`srun` terminated by the successful head process; both
`MUON_DAPO_AB_DONE job=14285256 rc=0` and
`MUON_DAPO_JOB_DONE job=14285256 rc=0` are present in the top-level log.

The immutable CW source logs remain at the paths recorded in
`source_manifest.sha256`; the manifest includes their SHA-256 digests and byte
sizes. Reproduce the analysis with:

```bash
python docs/runs/analyze_muon_dapo_rewards.py \
  --muon docs/runs/evidence/muon_dapo_14285256/muon_rewards.jsonl \
  --adam docs/runs/evidence/muon_dapo_14285256/adam_rewards.jsonl \
  --window-size 10 \
  --output /tmp/muon_dapo_14285256_analysis
```
