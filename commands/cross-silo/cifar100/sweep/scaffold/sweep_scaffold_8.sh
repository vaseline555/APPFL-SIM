#!/bin/bash -l

# CIFAR-100 Dirichlet Non-IID
for lr_decay in 0.935 0.93 0.9275 0.925; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/cross-silo/cifar100/scaffold.yaml \
        logging.configs.wandb_entity=vaseline555 train.local_epochs=8 \
          optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
            experiment.name=GALE_CIFAR100_SWEEP logging.name="sweep_scaffold_8_${lr_decay}" &
done
wait
