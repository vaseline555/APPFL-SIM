#!/bin/bash -l

set -euo pipefail
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Original loop/template
# # femnist sweep for fedadam with E=8
# for lr_decay in 0.935 0.93 0.9275 0.925; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-device/femnist/fedadam.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=fedadam \
#       "logging.configs.wandb_tags=cross-device,femnist,fedadam,sweep" \
#       train.local_epochs=8 \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=0.01 \
#       optimizer.lr_decay.gamma=$lr_decay \
#       experiment.name=GALE_FEMNIST_SWEEP \
#       logging.name="sweep_femnist_fedadam_8_${lr_decay}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedadam.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedadam logging.configs.wandb_tags=cross-device,femnist,fedadam,sweep train.local_epochs=8 optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.935 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_fedadam_8_0.935
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedadam.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedadam logging.configs.wandb_tags=cross-device,femnist,fedadam,sweep train.local_epochs=8 optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.93 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_fedadam_8_0.93
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedadam.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedadam logging.configs.wandb_tags=cross-device,femnist,fedadam,sweep train.local_epochs=8 optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.9275 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_fedadam_8_0.9275
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedadam.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedadam logging.configs.wandb_tags=cross-device,femnist,fedadam,sweep train.local_epochs=8 optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.925 experiment.name=GALE_FEMNIST_SWEEP logging.name=sweep_femnist_fedadam_8_0.925
