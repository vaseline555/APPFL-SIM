#!/bin/bash -l

set -euo pipefail

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# 20newsgroups sweep for gale_prox_c
for lr_decay in 0.95 0.96 0.97 0.98 0.99; do
  for ridge_alpha in 0.1 1.0 10.0; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/cross-silo/20newsgroups/gale_prox_c.yaml \
        "logging.configs.wandb_entity=${WANDB_ENTITY}" \
        "logging.configs.wandb_mode=${WANDB_MODE}" \
        logging.configs.wandb_group=gale_prox_c \
        "logging.configs.wandb_tags=cross-silo,20newsgroups,gale_prox_c,sweep" \
        optimizer.lr_decay.enable=true \
        optimizer.lr=5e-05 \
        optimizer.lr_decay.gamma=$lr_decay \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.01 \
        algorithm.scheduler_kwargs.ridge_alpha=$ridge_alpha \
        algorithm.scheduler_kwargs.reward_scale=10 \
        algorithm.scheduler_kwargs.contexts="[l,d]" \
        algorithm.trainer_kwargs.mu=0.3 \
        experiment.name=GALE_20NEWSGROUPS_SWEEP \
        logging.name="sweep_20newsgroups_gale_prox_c_${lr_decay}_${ridge_alpha}" &
  done
  wait
done
