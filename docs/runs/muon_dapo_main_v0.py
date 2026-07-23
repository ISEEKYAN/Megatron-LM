#!/usr/bin/env python3
"""Hydra entry for VERL's in-tree legacy RayPPOTrainer.

The pinned VERL checkout carries both trainer generations. Its v1 entry imports
the separately distributed ``transfer_queue`` package, which is not present in
the proven DAPO image. The in-tree v0 runner is the same RayPPOTrainer path used
by the established Qwen3.5 DAPO harness and needs no compatibility stub.
"""

import hydra

from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_v0 import TaskRunner


@hydra.main(
    config_path="../../../verl_muon_sft/verl/trainer/config",
    config_name="ppo_trainer",
    version_base=None,
)
def main(config):
    run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()
