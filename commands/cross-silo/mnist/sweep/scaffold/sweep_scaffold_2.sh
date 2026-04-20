#!/bin/bash -l

set -euo pipefail

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# mnist sweep for scaffold with E=2
for lr_decay in 0.98 0.975 0.97 0.965; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/mnist/scaffold.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=scaffold \
      "logging.configs.wandb_tags=cross-silo,mnist,scaffold,sweep" \
      train.local_epochs=2 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.01 \
      optimizer.lr_decay.gamma=$lr_decay \
      experiment.name=GALE_MNIST_SWEEP \
      logging.name="sweep_mnist_scaffold_2_${lr_decay}" &
done
wait
