#!/bin/bash -l

# CIFAR-100 Dirichlet Non-IID
for lr_decay in 0.999 0.995 0.99 0.985; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/cifar100/fedavg.yaml \
      logging.configs.wandb_entity=vaseline555 train.local_epochs=1 \
        optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
          experiment.name=GALE_CIFAR100_SWEEP logging.name="sweep_fedavg_1_${lr_decay}" &
done
wait
