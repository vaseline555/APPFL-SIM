#!/bin/bash -l

set -euo pipefail
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

for lr_decay in 0.95 0.96 0.97 0.98 0.99; do
  for ridge_alpha in 0.1 1.0 10.0; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/cross-device/londonsmartmeter/gale_prox_r.yaml \
      logging.backend=file \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=gale_prox_r \
      "logging.configs.wandb_tags=cross-device,londonsmartmeter,gale_prox_r,sweep" \
      optimizer.lr=0.001 \
      "optimizer.lr_decay.gamma=${lr_decay}" \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
      algorithm.scheduler_kwargs.exploration_beta=0.01 \
      "algorithm.scheduler_kwargs.ridge_alpha=${ridge_alpha}" \
      algorithm.scheduler_kwargs.reward_scale=10 \
      "algorithm.scheduler_kwargs.contexts=[l,d]" \
      algorithm.trainer_kwargs.mu=0.3 \
      experiment.name=GALE_LONDONSMARTMETER_SWEEP \
      "logging.name=sweep_londonsmartmeter_gale_prox_r_${lr_decay}_${ridge_alpha}" &
  done
  wait
done
