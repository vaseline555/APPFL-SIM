#!/bin/bash -l

set -euo pipefail

# Original loop/template
# # CIFAR-100 Dirichlet Non-IID
# # for mu in 0.05 0.07 0.1 0.2 0.3; do
# #     python -m appfl_sim.runner \
# #       --config appfl_sim/config/cross-device/tinyimagenet/fedprox.yaml \
# #         logging.configs.wandb_entity=vaseline555 train.local_epochs=2 \
# #          optimizer.lr=0.001 optimizer.lr_decay.gamma=0.985 \
# #             algorithm.trainer_kwargs.mu=$mu \
# #               experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="fedprox_2_${mu}" &
# # done
# # wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedprox.yaml logging.configs.wandb_entity=vaseline555 train.local_epochs=2 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.985 algorithm.trainer_kwargs.mu=0.05 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=fedprox_2_0.05
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedprox.yaml logging.configs.wandb_entity=vaseline555 train.local_epochs=2 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.985 algorithm.trainer_kwargs.mu=0.07 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=fedprox_2_0.07
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedprox.yaml logging.configs.wandb_entity=vaseline555 train.local_epochs=2 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.985 algorithm.trainer_kwargs.mu=0.1 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=fedprox_2_0.1
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedprox.yaml logging.configs.wandb_entity=vaseline555 train.local_epochs=2 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.985 algorithm.trainer_kwargs.mu=0.2 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=fedprox_2_0.2
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedprox.yaml logging.configs.wandb_entity=vaseline555 train.local_epochs=2 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.985 algorithm.trainer_kwargs.mu=0.3 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=fedprox_2_0.3
