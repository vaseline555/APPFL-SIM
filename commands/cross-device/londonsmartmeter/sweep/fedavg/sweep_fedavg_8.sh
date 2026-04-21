#!/bin/bash -l

set -euo pipefail
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Original loop/template
# # londonsmartmeter sweep for fedavg with E=8
# for lr_decay in 0.965 0.96 0.955 0.95; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-device/londonsmartmeter/fedavg.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=fedavg \
#       "logging.configs.wandb_tags=cross-device,londonsmartmeter,fedavg,sweep" \
#       train.local_epochs=8 \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=0.001 \
#       optimizer.lr_decay.gamma=$lr_decay \
#       experiment.name=GALE_LONDONSMARTMETER_SWEEP \
#       logging.name="sweep_londonsmartmeter_fedavg_8_${lr_decay}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/londonsmartmeter/fedavg.yaml logging.backend=file logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-device,londonsmartmeter,fedavg,sweep train.local_epochs=8 optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.965 experiment.name=GALE_LONDONSMARTMETER_SWEEP logging.name=sweep_londonsmartmeter_fedavg_8_0.965
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/londonsmartmeter/fedavg.yaml logging.backend=file logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-device,londonsmartmeter,fedavg,sweep train.local_epochs=8 optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.96 experiment.name=GALE_LONDONSMARTMETER_SWEEP logging.name=sweep_londonsmartmeter_fedavg_8_0.96
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/londonsmartmeter/fedavg.yaml logging.backend=file logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-device,londonsmartmeter,fedavg,sweep train.local_epochs=8 optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.955 experiment.name=GALE_LONDONSMARTMETER_SWEEP logging.name=sweep_londonsmartmeter_fedavg_8_0.955
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/londonsmartmeter/fedavg.yaml logging.backend=file logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-device,londonsmartmeter,fedavg,sweep train.local_epochs=8 optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.95 experiment.name=GALE_LONDONSMARTMETER_SWEEP logging.name=sweep_londonsmartmeter_fedavg_8_0.95
