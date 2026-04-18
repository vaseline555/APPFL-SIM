#!/bin/bash -l

# CIFAR-100 Dirichlet Non-IID
for lr_decay in 0.995 0.99 0.985 0.98; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/cifar100/fedavg.yaml \
      logging.configs.wandb_entity=vaseline555 train.local_epochs=2 \
        optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
          experiment.name=GALE_CIFAR100_SWEEP logging.name="sweep_fedavg_2_${lr_decay}" &
done
wait
