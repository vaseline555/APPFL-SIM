#!/bin/bash -l

set -euo pipefail
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Original loop/template
# # femnist sweep for fedavg with E=2
# for lr_decay in 0.995 0.99 0.985 0.98; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-device/femnist/fedavg.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=fedavg \
#       "logging.configs.wandb_tags=cross-device,femnist,fedavg,sweep" \
#       train.local_epochs=2 \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=0.001 \
#       optimizer.lr_decay.gamma=$lr_decay \
#       experiment.name=GALE_FEMNIST_SWEEP \
#       logging.name="sweep_femnist_fedavg_2_${lr_decay}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedavg.yaml logging.backend=file logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-device,femnist,fedavg,sweep train.local_epochs=2 optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_fedavg_2_0.995
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedavg.yaml logging.backend=file logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-device,femnist,fedavg,sweep train.local_epochs=2 optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_fedavg_2_0.99
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedavg.yaml logging.backend=file logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-device,femnist,fedavg,sweep train.local_epochs=2 optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.985 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_fedavg_2_0.985
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedavg.yaml logging.backend=file logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-device,femnist,fedavg,sweep train.local_epochs=2 optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_fedavg_2_0.98
