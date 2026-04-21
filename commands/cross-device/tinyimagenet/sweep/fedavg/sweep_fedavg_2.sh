#!/bin/bash -l

set -euo pipefail

# Original loop/template
# # TinyImageNet Dirichlet Non-IID
# # for lr_decay in 0.995 0.99 0.985 0.98; do
# #   python -m appfl_sim.runner \
# #     --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
# #       train.local_epochs=2 \
# #         optimizer.lr=0.1 optimizer.lr_decay.gamma=$lr_decay \
# #           experiment.name=GALE_TINYIMAGENET_SWEEP \
# #             logging.name="sweep_fedavg_2_${lr_decay}" &
# # done
# # wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml logging.backend=file train.local_epochs=2 optimizer.lr=0.1 optimizer.lr_decay.gamma=0.995 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedavg_2_0.995
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml logging.backend=file train.local_epochs=2 optimizer.lr=0.1 optimizer.lr_decay.gamma=0.99 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedavg_2_0.99
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml logging.backend=file train.local_epochs=2 optimizer.lr=0.1 optimizer.lr_decay.gamma=0.985 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedavg_2_0.985
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml logging.backend=file train.local_epochs=2 optimizer.lr=0.1 optimizer.lr_decay.gamma=0.98 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedavg_2_0.98
