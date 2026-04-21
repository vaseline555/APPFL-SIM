#!/bin/bash -l

set -euo pipefail
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Original loop/template
# # londonsmartmeter sweep for gale_avg
# for lr_decay in 0.98 0.99 0.995; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-device/londonsmartmeter/gale_avg.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=gale_avg \
#       "logging.configs.wandb_tags=cross-device,londonsmartmeter,gale_avg,sweep" \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=0.001 \
#       optimizer.lr_decay.gamma=$lr_decay \
#       algorithm.scheduler_kwargs.discount_gamma=0.80 \
#       algorithm.scheduler_kwargs.exploration_alpha=0.001 \
#       algorithm.scheduler_kwargs.reward_scale=1 \
#       experiment.name=GALE_LONDONSMARTMETER_SWEEP \
#       logging.name="sweep_londonsmartmeter_gale_avg_80_0.001_${lr_decay}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/londonsmartmeter/gale_avg.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=gale_avg logging.configs.wandb_tags=cross-device,londonsmartmeter,gale_avg,sweep optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 experiment.name=GALE_LONDONSMARTMETER_SWEEP logging.name=sweep_londonsmartmeter_gale_avg_80_0.001_0.98
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/londonsmartmeter/gale_avg.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=gale_avg logging.configs.wandb_tags=cross-device,londonsmartmeter,gale_avg,sweep optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 experiment.name=GALE_LONDONSMARTMETER_SWEEP logging.name=sweep_londonsmartmeter_gale_avg_80_0.001_0.99
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/londonsmartmeter/gale_avg.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=gale_avg logging.configs.wandb_tags=cross-device,londonsmartmeter,gale_avg,sweep optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 experiment.name=GALE_LONDONSMARTMETER_SWEEP logging.name=sweep_londonsmartmeter_gale_avg_80_0.001_0.995
