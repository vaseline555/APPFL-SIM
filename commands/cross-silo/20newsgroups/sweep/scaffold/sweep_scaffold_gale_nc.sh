#!/bin/bash -l

set -euo pipefail
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Original loop/template
# # 20newsgroups sweep for gale_scaffold
# for lr_decay in 0.95 0.96 0.97 0.98 0.99; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-silo/20newsgroups/gale_scaffold.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=gale_scaffold \
#       "logging.configs.wandb_tags=cross-silo,20newsgroups,gale_scaffold,sweep" \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=5e-05 \
#       optimizer.lr_decay.gamma=$lr_decay \
#       algorithm.scheduler_kwargs.discount_gamma=0.80 \
#       algorithm.scheduler_kwargs.exploration_alpha=0.0005 \
#       algorithm.scheduler_kwargs.reward_scale=10 \
#       experiment.name=GALE_20NEWSGROUPS_SWEEP \
#       logging.name="sweep_20newsgroups_gale_scaffold_${lr_decay}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/gale_scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=gale_scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,gale_scaffold,sweep optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.95 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.0005 algorithm.scheduler_kwargs.reward_scale=10 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_gale_scaffold_0.95
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/gale_scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=gale_scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,gale_scaffold,sweep optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.96 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.0005 algorithm.scheduler_kwargs.reward_scale=10 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_gale_scaffold_0.96
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/gale_scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=gale_scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,gale_scaffold,sweep optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.97 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.0005 algorithm.scheduler_kwargs.reward_scale=10 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_gale_scaffold_0.97
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/gale_scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=gale_scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,gale_scaffold,sweep optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.98 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.0005 algorithm.scheduler_kwargs.reward_scale=10 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_gale_scaffold_0.98
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/gale_scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=gale_scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,gale_scaffold,sweep optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.99 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.0005 algorithm.scheduler_kwargs.reward_scale=10 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_gale_scaffold_0.99
