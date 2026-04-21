#!/bin/bash -l

set -euo pipefail

# Original loop/template
# # TinyImageNet Dirichlet Non-IID
# # for lr_decay in 0.98 0.97 0.96 0.95; do
# #   python -m appfl_sim.runner \
# #     --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
# #       train.local_epochs=4 \
# #         optimizer.lr=0.1 optimizer.lr_decay.gamma=$lr_decay \
# #           experiment.name=GALE_TINYIMAGENET_SWEEP \
# #             logging.name="sweep_fedavg_4_${lr_decay}" &
# # done
# # wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml logging.backend=file train.local_epochs=4 optimizer.lr=0.1 optimizer.lr_decay.gamma=0.98 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedavg_4_0.98
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml logging.backend=file train.local_epochs=4 optimizer.lr=0.1 optimizer.lr_decay.gamma=0.97 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedavg_4_0.97
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml logging.backend=file train.local_epochs=4 optimizer.lr=0.1 optimizer.lr_decay.gamma=0.96 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedavg_4_0.96
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml logging.backend=file train.local_epochs=4 optimizer.lr=0.1 optimizer.lr_decay.gamma=0.95 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedavg_4_0.95
