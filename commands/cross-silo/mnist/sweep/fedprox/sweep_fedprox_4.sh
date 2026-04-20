#!/bin/bash -l

set -euo pipefail

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# mnist sweep for fedprox with E=4
for mu in 0.05 0.07 0.1 0.2 0.3; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/mnist/fedprox.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=fedprox \
      "logging.configs.wandb_tags=cross-silo,mnist,fedprox,sweep" \
      train.local_epochs=4 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.01 \
      optimizer.lr_decay.gamma=0.96 \
      algorithm.trainer_kwargs.mu=$mu \
      experiment.name=GALE_MNIST_SWEEP \
      logging.name="sweep_mnist_fedprox_4_${mu}" &
done
wait
