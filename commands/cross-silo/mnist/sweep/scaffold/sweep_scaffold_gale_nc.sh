#!/bin/bash -l

set -euo pipefail

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# mnist sweep for gale_scaffold
for lr_decay in 0.95 0.96 0.97 0.98 0.99; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/mnist/gale_scaffold.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=gale_scaffold \
      "logging.configs.wandb_tags=cross-silo,mnist,gale_scaffold,sweep" \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.01 \
      optimizer.lr_decay.gamma=$lr_decay \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
      algorithm.scheduler_kwargs.exploration_alpha=0.0005 \
      algorithm.scheduler_kwargs.reward_scale=10 \
      experiment.name=GALE_MNIST_SWEEP \
      logging.name="sweep_mnist_gale_scaffold_${lr_decay}" &
done
wait
