#!/bin/bash -l

set -euo pipefail
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Original loop/template
# # 20newsgroups sweep for fedprox with E=2
# for mu in 0.05 0.07 0.1 0.2 0.3; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-silo/20newsgroups/fedprox.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=fedprox \
#       "logging.configs.wandb_tags=cross-silo,20newsgroups,fedprox,sweep" \
#       train.local_epochs=2 \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=5e-05 \
#       optimizer.lr_decay.gamma=0.985 \
#       algorithm.trainer_kwargs.mu=$mu \
#       experiment.name=GALE_20NEWSGROUPS_SWEEP \
#       logging.name="sweep_20newsgroups_fedprox_2_${mu}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/fedprox.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedprox logging.configs.wandb_tags=cross-silo,20newsgroups,fedprox,sweep train.local_epochs=2 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.985 algorithm.trainer_kwargs.mu=0.05 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_fedprox_2_0.05
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/fedprox.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedprox logging.configs.wandb_tags=cross-silo,20newsgroups,fedprox,sweep train.local_epochs=2 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.985 algorithm.trainer_kwargs.mu=0.07 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_fedprox_2_0.07
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/fedprox.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedprox logging.configs.wandb_tags=cross-silo,20newsgroups,fedprox,sweep train.local_epochs=2 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.985 algorithm.trainer_kwargs.mu=0.1 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_fedprox_2_0.1
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/fedprox.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedprox logging.configs.wandb_tags=cross-silo,20newsgroups,fedprox,sweep train.local_epochs=2 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.985 algorithm.trainer_kwargs.mu=0.2 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_fedprox_2_0.2
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/fedprox.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=fedprox logging.configs.wandb_tags=cross-silo,20newsgroups,fedprox,sweep train.local_epochs=2 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.985 algorithm.trainer_kwargs.mu=0.3 experiment.name=GALE_20NEWSGROUPS_SWEEP logging.name=sweep_20newsgroups_fedprox_2_0.3
