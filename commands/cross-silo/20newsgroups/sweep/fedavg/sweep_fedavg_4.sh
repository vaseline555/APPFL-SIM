#!/bin/bash -l

set -euo pipefail
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Original loop/template
# # 20newsgroups sweep for fedavg with E=4
# for lr_decay in 0.98 0.97 0.96 0.95; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-silo/20newsgroups/fedavg.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=fedavg \
#       "logging.configs.wandb_tags=cross-silo,20newsgroups,fedavg,sweep" \
#       train.local_epochs=4 \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=5e-05 \
#       optimizer.lr_decay.gamma=$lr_decay \
#       experiment.name=GALE_20NEWSGROUPS_SWEEP \
#       logging.name="sweep_20newsgroups_fedavg_4_${lr_decay}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/fedavg.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-silo,20newsgroups,fedavg,sweep train.local_epochs=4 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.98 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_fedavg_4_0.98
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/fedavg.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-silo,20newsgroups,fedavg,sweep train.local_epochs=4 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.97 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_fedavg_4_0.97
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/fedavg.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-silo,20newsgroups,fedavg,sweep train.local_epochs=4 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.96 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_fedavg_4_0.96
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/fedavg.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-silo,20newsgroups,fedavg,sweep train.local_epochs=4 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.95 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_fedavg_4_0.95
