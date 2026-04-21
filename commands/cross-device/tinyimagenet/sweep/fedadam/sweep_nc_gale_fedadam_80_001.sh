#!/bin/bash -l

set -euo pipefail

# Original loop/template
# # for lr_decay in 0.93 0.95 0.97 0.99; do
# #   python -m appfl_sim.runner \
# #     --config appfl_sim/config/cross-device/tinyimagenet/gale_adam.yaml \
# #       optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
# #         algorithm.scheduler_kwargs.discount_gamma=0.80 \
# #           algorithm.scheduler_kwargs.exploration_alpha=0.001 \
# #             algorithm.scheduler_kwargs.reward_scale=1 \
# #               experiment.name=GALE_TINYIMAGENET_SWEEP \
# #                 logging.name="sweep_gale_adam_80_0.001_${lr_decay}" &
# # done
# # wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/gale_adam.yaml optimizer.lr=0.001 optimizer.lr_decay.gamma=0.93 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_gale_adam_80_0.001_0.93
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/gale_adam.yaml optimizer.lr=0.001 optimizer.lr_decay.gamma=0.95 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_gale_adam_80_0.001_0.95
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/gale_adam.yaml optimizer.lr=0.001 optimizer.lr_decay.gamma=0.97 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_gale_adam_80_0.001_0.97
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/gale_adam.yaml optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_alpha=0.001 algorithm.scheduler_kwargs.reward_scale=1 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_gale_adam_80_0.001_0.99
