#!/bin/bash -l

# Run
for lr_decay in 0.95 0.96 0.97 0.98 0.99; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/cifar100/gale_scaffold.yaml \
      logging.configs.wandb_entity=vaseline555 \
        optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
          algorithm.scheduler_kwargs.discount_gamma=0.80 \
            algorithm.scheduler_kwargs.exploration_alpha=0.0005 \
              algorithm.scheduler_kwargs.reward_scale=1 \
                experiment.name=GALE_CIFAR100_SWEEP logging.name="sweep_scaffold_gale_nc_${lr_decay}" &
done
wait
