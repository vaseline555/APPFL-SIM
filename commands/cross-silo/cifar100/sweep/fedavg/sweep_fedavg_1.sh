#!/bin/bash -l

# Original loop/template
# # CIFAR-100 Dirichlet Non-IID
# for lr_decay in 0.999 0.995 0.99 0.985; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-silo/cifar100/fedavg.yaml \
#       logging.configs.wandb_entity=vaseline555 train.local_epochs=1 \
#         optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
#           experiment.name=GALE_CIFAR100_SWEEP logging.name="sweep_fedavg_1_${lr_decay}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/cifar100/fedavg.yaml logging.configs.wandb_entity=vaseline555 train.local_epochs=1 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.999 experiment.name=GALE_CIFAR100_SWEEP logging.name=sweep_fedavg_1_0.999
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/cifar100/fedavg.yaml logging.configs.wandb_entity=vaseline555 train.local_epochs=1 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 experiment.name=GALE_CIFAR100_SWEEP logging.name=sweep_fedavg_1_0.995
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/cifar100/fedavg.yaml logging.configs.wandb_entity=vaseline555 train.local_epochs=1 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 experiment.name=GALE_CIFAR100_SWEEP logging.name=sweep_fedavg_1_0.99
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/cifar100/fedavg.yaml logging.configs.wandb_entity=vaseline555 train.local_epochs=1 optimizer.lr=0.001 optimizer.lr_decay.gamma=0.985 experiment.name=GALE_CIFAR100_SWEEP logging.name=sweep_fedavg_1_0.985
