#!/bin/bash -l

set -euo pipefail

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# 20newsgroups sweep for fedprox with E=8
for mu in 0.05 0.07 0.1 0.2 0.3; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/20newsgroups/fedprox.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=fedprox \
      "logging.configs.wandb_tags=cross-silo,20newsgroups,fedprox,sweep" \
      train.local_epochs=8 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=5e-05 \
      optimizer.lr_decay.gamma=0.95 \
      algorithm.trainer_kwargs.mu=$mu \
      experiment.name=GALE_20NEWSGROUPS_SWEEP \
      logging.name="sweep_20newsgroups_fedprox_8_${mu}" &
done
wait
