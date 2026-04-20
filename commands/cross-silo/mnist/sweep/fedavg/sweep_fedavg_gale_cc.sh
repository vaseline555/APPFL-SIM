#!/bin/bash -l

set -euo pipefail

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# mnist sweep for gale_avg_c
for lr_decay in 0.95 0.96 0.97 0.98 0.99; do
  for ridge_alpha in 0.1 1.0 10.0; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/cross-silo/mnist/gale_avg_c.yaml \
        "logging.configs.wandb_entity=${WANDB_ENTITY}" \
        "logging.configs.wandb_mode=${WANDB_MODE}" \
        logging.configs.wandb_group=gale_avg_c \
        "logging.configs.wandb_tags=cross-silo,mnist,gale_avg_c,sweep" \
        optimizer.lr_decay.enable=true \
        optimizer.lr=0.01 \
        optimizer.lr_decay.gamma=$lr_decay \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.01 \
        algorithm.scheduler_kwargs.ridge_alpha=$ridge_alpha \
        algorithm.scheduler_kwargs.reward_scale=10 \
        algorithm.scheduler_kwargs.contexts="[l,d]" \
        experiment.name=GALE_MNIST_SWEEP \
        logging.name="sweep_mnist_gale_avg_c_${lr_decay}_${ridge_alpha}" &
  done
  wait
done
