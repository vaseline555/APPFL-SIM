#!/bin/bash -l

set -euo pipefail
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Original loop/template
# # londonsmartmeter sweep for fedprox with E=1
# for mu in 0.05 0.07 0.1 0.2 0.3; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-device/londonsmartmeter/fedprox.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=fedprox \
#       "logging.configs.wandb_tags=cross-device,londonsmartmeter,fedprox,sweep" \
#       train.local_epochs=1 \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=0.001 \
#       optimizer.lr_decay.gamma=0.999 \
#       algorithm.trainer_kwargs.mu=$mu \
#       experiment.name=GALE_LONDONSMARTMETER_SWEEP \
#       logging.name="sweep_londonsmartmeter_fedprox_1_${mu}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/londonsmartmeter/fedprox.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedprox logging.configs.wandb_tags=cross-device,londonsmartmeter,fedprox,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.999 algorithm.trainer_kwargs.mu=0.05 experiment.name=GALE_LONDONSMARTMETER_SWEEP logging.name=sweep_londonsmartmeter_fedprox_1_0.05
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/londonsmartmeter/fedprox.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedprox logging.configs.wandb_tags=cross-device,londonsmartmeter,fedprox,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.999 algorithm.trainer_kwargs.mu=0.07 experiment.name=GALE_LONDONSMARTMETER_SWEEP logging.name=sweep_londonsmartmeter_fedprox_1_0.07
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/londonsmartmeter/fedprox.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedprox logging.configs.wandb_tags=cross-device,londonsmartmeter,fedprox,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.999 algorithm.trainer_kwargs.mu=0.1 experiment.name=GALE_LONDONSMARTMETER_SWEEP logging.name=sweep_londonsmartmeter_fedprox_1_0.1
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/londonsmartmeter/fedprox.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedprox logging.configs.wandb_tags=cross-device,londonsmartmeter,fedprox,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.999 algorithm.trainer_kwargs.mu=0.2 experiment.name=GALE_LONDONSMARTMETER_SWEEP logging.name=sweep_londonsmartmeter_fedprox_1_0.2
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/londonsmartmeter/fedprox.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedprox logging.configs.wandb_tags=cross-device,londonsmartmeter,fedprox,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=0.001 optimizer.lr_decay.gamma=0.999 algorithm.trainer_kwargs.mu=0.3 experiment.name=GALE_LONDONSMARTMETER_SWEEP logging.name=sweep_londonsmartmeter_fedprox_1_0.3
