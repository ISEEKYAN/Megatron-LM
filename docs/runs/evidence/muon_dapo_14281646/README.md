# Qwen3-30B-A3B DAPO reward A/B

Slurm job `14281646` ran on 16 H100 GPUs (two nodes) and completed with
`ExitCode=0:0`. It ran the Muon and AdamW arms sequentially from the same
Qwen3-30B-A3B checkpoint with seed 42, the DAPO reward manager, a batch size
of 16, four responses per prompt, and five optimizer steps. Both arms used
LR `1e-5`, constant scheduling, BF16 model parameters, and FP32 optimizer
master/state tensors. The optimizer algorithm was the experimental variable.

The observed `critic/rewards/mean` curves were:

| Step | Muon | AdamW |
| ---: | ---: | ---: |
| 1 | -1.00000 | -1.00000 |
| 2 | -0.93750 | -0.96875 |
| 3 | -1.00000 | -1.00000 |
| 4 | -0.96875 | -0.96875 |
| 5 | -0.90625 | -0.96875 |

Muon gained `+0.09375` from the first to the last step; AdamW gained
`+0.03125`. Thus Muon produced a positive reward gain and its gain was
`+0.0625` above AdamW in this matched short-horizon run.

The machine-readable values are in `reward_curves.json`, and
`reward_curves.svg` is the corresponding dual curve. The immutable raw logs
remain on CW under:

```
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/runtime/muon-dapo-reward-ab-30b/job-14281646/
```

This is a single-seed, five-step hard-gate experiment. It demonstrates that
the real Muon DAPO path updates reward and is not worse than the matched AdamW
arm over this window; it is not a claim about long-run convergence variance.
