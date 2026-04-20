#!/bin/bash -l

set -euo pipefail

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# 20newsgroups sweep for fedavg with E=8
for lr_decay in 0.965 0.96 0.955 0.95; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/20newsgroups/fedavg.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=fedavg \
      "logging.configs.wandb_tags=cross-silo,20newsgroups,fedavg,sweep" \
      train.local_epochs=8 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=5e-05 \
      optimizer.lr_decay.gamma=$lr_decay \
      experiment.name=GALE_20NEWSGROUPS_SWEEP \
      logging.name="sweep_20newsgroups_fedavg_8_${lr_decay}" &
done
wait
