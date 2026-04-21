#!/bin/bash -l

set -euo pipefail
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Original loop/template
# # femnist sweep for fedadam with E=1
# for lr_decay in 0.999 0.995 0.99 0.985; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-device/femnist/fedadam.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=fedadam \
#       "logging.configs.wandb_tags=cross-device,femnist,fedadam,sweep" \
#       train.local_epochs=1 \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=0.01 \
#       optimizer.lr_decay.gamma=$lr_decay \
#       experiment.name=GALE_FEMNIST_SWEEP \
#       logging.name="sweep_femnist_fedadam_1_${lr_decay}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedadam.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedadam logging.configs.wandb_tags=cross-device,femnist,fedadam,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.999 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_fedadam_1_0.999
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedadam.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedadam logging.configs.wandb_tags=cross-device,femnist,fedadam,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.995 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_fedadam_1_0.995
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedadam.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedadam logging.configs.wandb_tags=cross-device,femnist,fedadam,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.99 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_fedadam_1_0.99
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedadam.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedadam logging.configs.wandb_tags=cross-device,femnist,fedadam,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.985 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_fedadam_1_0.985
