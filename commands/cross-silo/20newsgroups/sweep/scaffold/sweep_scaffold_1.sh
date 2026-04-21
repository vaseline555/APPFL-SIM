#!/bin/bash -l

set -euo pipefail
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Original loop/template
# # 20newsgroups sweep for scaffold with E=1
# for lr_decay in 0.999 0.995 0.99 0.985; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=scaffold \
#       "logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,sweep" \
#       train.local_epochs=1 \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=5e-05 \
#       optimizer.lr_decay.gamma=$lr_decay \
#       experiment.name=GALE_20NEWSGROUPS_SWEEP \
#       logging.name="sweep_20newsgroups_scaffold_1_${lr_decay}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.999 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_scaffold_1_0.999
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.995 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_scaffold_1_0.995
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.99 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_scaffold_1_0.99
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.985 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_scaffold_1_0.985
