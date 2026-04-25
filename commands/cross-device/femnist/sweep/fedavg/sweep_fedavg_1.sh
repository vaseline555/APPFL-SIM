#!/bin/bash -l


set -euo pipefail

cd "${PBS_O_WORKDIR:-$PWD}"

module use /soft/modulefiles
module load conda
conda activate base

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Original loop/template
# # femnist sweep for fedavg with E=1
# for lr_decay in 0.999 0.995 0.99 0.985; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-device/femnist/fedavg.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=fedavg \
#       "logging.configs.wandb_tags=cross-device,femnist,fedavg,sweep" \
#       train.local_epochs=1 \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=0.001 \
#       optimizer.lr_decay.gamma=$lr_decay \
#       experiment.name=GALE_SWEEP \
#       logging.name="sweep_femnist_fedavg_1_${lr_decay}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES=0 python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedavg.yaml logging.backend=file logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-device,femnist,fedavg,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=0.1 optimizer.lr_decay.gamma=0.9975 experiment.name=GALE_SWEEP logging.name=sweep_femnist_fedavg_1_0.1 &
CUDA_VISIBLE_DEVICES=1 python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedavg.yaml logging.backend=file logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-device,femnist,fedavg,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.9975 experiment.name=GALE_SWEEP logging.name=sweep_femnist_fedavg_1_0.01 &
CUDA_VISIBLE_DEVICES=2 python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedavg.yaml logging.backend=file logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-device,femnist,fedavg,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=0.316 optimizer.lr_decay.gamma=0.9975 experiment.name=GALE_SWEEP logging.name=sweep_femnist_fedavg_1_0.316 &
CUDA_VISIBLE_DEVICES=3 python -m appfl_sim.runner --config appfl_sim/config/cross-device/femnist/fedavg.yaml logging.backend=file logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedavg logging.configs.wandb_tags=cross-device,femnist,fedavg,sweep train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=0.0316 optimizer.lr_decay.gamma=0.9975 experiment.name=GALE_SWEEP logging.name=sweep_femnist_fedavg_1_0.0316 &
