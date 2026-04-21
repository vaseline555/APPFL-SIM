#!/bin/bash -l

set -euo pipefail

# Original loop/template
# # TinyImageNet Dirichlet Non-IID
# # for lr_decay in 0.98 0.97 0.96 0.95; do
# #   python -m appfl_sim.runner \
# #     --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml \
# #       train.local_epochs=4 \
# #         optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
# #           experiment.name=GALE_TINYIMAGENET_SWEEP \
# #             logging.name="sweep_fedadam_4_${lr_decay}" &
# # done
# # wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml train.local_epochs=4 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedadam_4_0.98
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml train.local_epochs=4 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.97 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedadam_4_0.97
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml train.local_epochs=4 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.96 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedadam_4_0.96
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml train.local_epochs=4 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.95 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedadam_4_0.95
