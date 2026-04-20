#!/bin/bash -l

set -euo pipefail

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# mnist sweep for scaffold with E=1
for lr_decay in 0.999 0.995 0.99 0.985; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/mnist/scaffold.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=scaffold \
      "logging.configs.wandb_tags=cross-silo,mnist,scaffold,sweep" \
      train.local_epochs=1 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.01 \
      optimizer.lr_decay.gamma=$lr_decay \
      experiment.name=GALE_MNIST_SWEEP \
      logging.name="sweep_mnist_scaffold_1_${lr_decay}" &
done
wait
