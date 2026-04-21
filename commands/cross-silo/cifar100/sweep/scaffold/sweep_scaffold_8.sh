#!/bin/bash -l

# Original loop/template
# # CIFAR-100 Dirichlet Non-IID
# for lr_decay in 0.935 0.93 0.9275 0.925; do
#     python -m appfl_sim.runner \
#       --config appfl_sim/config/cross-silo/cifar100/scaffold.yaml \
#         logging.configs.wandb_entity=vaseline555 train.local_epochs=8 \
#           optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
#             experiment.name=GALE_CIFAR100_SWEEP logging.name="sweep_scaffold_8_${lr_decay}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/cifar100/scaffold.yaml logging.configs.wandb_entity=vaseline555 train.local_epochs=8 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.935 experiment.name=GALE_CIFAR100_SWEEP logging.name=sweep_scaffold_8_0.935
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/cifar100/scaffold.yaml logging.configs.wandb_entity=vaseline555 train.local_epochs=8 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.93 experiment.name=GALE_CIFAR100_SWEEP logging.name=sweep_scaffold_8_0.93
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/cifar100/scaffold.yaml logging.configs.wandb_entity=vaseline555 train.local_epochs=8 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.9275 experiment.name=GALE_CIFAR100_SWEEP logging.name=sweep_scaffold_8_0.9275
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/cifar100/scaffold.yaml logging.configs.wandb_entity=vaseline555 train.local_epochs=8 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.925 experiment.name=GALE_CIFAR100_SWEEP logging.name=sweep_scaffold_8_0.925
