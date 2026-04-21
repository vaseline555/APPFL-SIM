#!/bin/bash -l

set -euo pipefail

# Original loop/template
# # for lr_decay in 0.95 0.96 0.97 0.98 0.99; do
# #   python -m appfl_sim.runner \
# #     --config appfl_sim/config/cross-device/tinyimagenet/gale_prox.yaml \
# #       logging.configs.wandb_entity=vaseline555 \
# #         optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
# #           algorithm.scheduler_kwargs.discount_gamma=0.80 \
# #             algorithm.scheduler_kwargs.exploration_alpha=0.001 \
# #               algorithm.scheduler_kwargs.reward_scale=1 \
# #                 algorithm.trainer_kwargs.mu=0.3 \
# #                   experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_80_0.001_${lr_decay}" &
# # done
# # wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/gale_prox.yaml logging.configs.wandb_entity=vaseline555 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.95 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 algorithm.trainer_kwargs.mu=0.3 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=GALE_PROX_80_0.001_0.95
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/gale_prox.yaml logging.configs.wandb_entity=vaseline555 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.96 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 algorithm.trainer_kwargs.mu=0.3 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=GALE_PROX_80_0.001_0.96
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/gale_prox.yaml logging.configs.wandb_entity=vaseline555 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.97 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 algorithm.trainer_kwargs.mu=0.3 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=GALE_PROX_80_0.001_0.97
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/gale_prox.yaml logging.configs.wandb_entity=vaseline555 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 algorithm.trainer_kwargs.mu=0.3 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=GALE_PROX_80_0.001_0.98
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/gale_prox.yaml logging.configs.wandb_entity=vaseline555 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 algorithm.trainer_kwargs.mu=0.3 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=GALE_PROX_80_0.001_0.99
