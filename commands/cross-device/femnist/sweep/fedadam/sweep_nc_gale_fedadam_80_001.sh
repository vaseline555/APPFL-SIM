#!/bin/bash -l

set -euo pipefail
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Original loop/template
# # femnist sweep for gale_adam
# for lr_decay in 0.93 0.95 0.97 0.99; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-device/femnist/gale_adam.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=gale_adam \
#       "logging.configs.wandb_tags=cross-device,femnist,gale_adam,sweep" \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=0.01 \
#       optimizer.lr_decay.gamma=$lr_decay \
#       algorithm.scheduler_kwargs.discount_gamma=0.80 \
#       algorithm.scheduler_kwargs.exploration_alpha=0.001 \
#       algorithm.scheduler_kwargs.reward_scale=1 \
#       experiment.name=GALE_FEMNIST_SWEEP \
#       logging.name="sweep_femnist_gale_adam_80_0.01_${lr_decay}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/gale_adam.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=gale_adam logging.configs.wandb_tags=cross-device,femnist,gale_adam,sweep optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.93 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_gale_adam_80_0.01_0.93
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/gale_adam.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=gale_adam logging.configs.wandb_tags=cross-device,femnist,gale_adam,sweep optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.95 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_gale_adam_80_0.01_0.95
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/gale_adam.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=gale_adam logging.configs.wandb_tags=cross-device,femnist,gale_adam,sweep optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.97 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_gale_adam_80_0.01_0.97
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/gale_adam.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=gale_adam logging.configs.wandb_tags=cross-device,femnist,gale_adam,sweep optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.99 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_gale_adam_80_0.01_0.99
