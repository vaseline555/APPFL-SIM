#!/bin/bash -l

# CIFAR-100 Dirichlet Non-IID
for mu in 0.05 0.07 0.1 0.2 0.3; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/cross-silo/cifar100/fedprox.yaml \
        logging.configs.wandb_entity=vaseline555 train.local_epochs=1 \
          optimizer.lr=0.001 optimizer.lr_decay.gamma=0.999 \
            algorithm.trainer_kwargs.mu=$mu \
              experiment.name=GALE_CIFAR100_SWEEP logging.name="sweep_fedprox_1_${mu}" &
done
wait
