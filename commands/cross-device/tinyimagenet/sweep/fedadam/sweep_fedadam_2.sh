#!/bin/bash -l

set -euo pipefail

# Original loop/template
# # TinyImageNet Dirichlet Non-IID
# # for lr_decay in 0.98 0.975 0.97 0.965; do
# #   python -m appfl_sim.runner \
# #     --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml \
# #       train.local_epochs=2 \
# #         optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
# #           experiment.name=GALE_TINYIMAGENET_SWEEP \
# #             logging.name="sweep_fedadam_2_${lr_decay}" &
# # done
# # wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml logging.backend=file train.local_epochs=2 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedadam_2_0.98
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml logging.backend=file train.local_epochs=2 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.975 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedadam_2_0.975
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml logging.backend=file train.local_epochs=2 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.97 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedadam_2_0.97
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml logging.backend=file train.local_epochs=2 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.965 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedadam_2_0.965
