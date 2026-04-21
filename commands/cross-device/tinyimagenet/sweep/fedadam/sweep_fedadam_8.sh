#!/bin/bash -l

set -euo pipefail

# Original loop/template
# # TinyImageNet Dirichlet Non-IID
# # for lr_decay in 0.935 0.93 0.9275 0.925; do
# #   python -m appfl_sim.runner \
# #     --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml \
# #       train.local_epochs=8 \
# #         optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
# #           experiment.name=GALE_TINYIMAGENET_SWEEP \
# #             logging.name="sweep_fedadam_8_${lr_decay}" &
# # done
# # wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml logging.backend=file train.local_epochs=8 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.935 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedadam_8_0.935
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml logging.backend=file train.local_epochs=8 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.93 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedadam_8_0.93
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml logging.backend=file train.local_epochs=8 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.9275 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedadam_8_0.9275
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml logging.backend=file train.local_epochs=8 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.925 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedadam_8_0.925
