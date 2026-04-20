#!/bin/bash -l

set -euo pipefail

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# 20newsgroups sweep for scaffold with E=4
for lr_decay in 0.98 0.97 0.96 0.95; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=scaffold \
      "logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,sweep" \
      train.local_epochs=4 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=5e-05 \
      optimizer.lr_decay.gamma=$lr_decay \
      experiment.name=GALE_20NEWSGROUPS_SWEEP \
      logging.name="sweep_20newsgroups_scaffold_4_${lr_decay}" &
done
wait
