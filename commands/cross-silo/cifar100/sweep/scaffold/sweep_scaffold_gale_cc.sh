#!/bin/bash -l

# Run
for lr_decay in 0.95 0.96 0.97 0.98 0.99; do
  for ridge_alpha in 0.1 1.0 10.0; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/cross-silo/cifar100/gale_scaffold_c.yaml \
        logging.configs.wandb_entity=vaseline555 \
          optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
            algorithm.scheduler_kwargs.discount_gamma=0.80 \
              algorithm.scheduler_kwargs.exploration_beta=0.01 \
                algorithm.scheduler_kwargs.ridge_alpha=$ridge_alpha algorithm.scheduler_kwargs.reward_scale=10 \
                  algorithm.scheduler_kwargs.contexts="[l,d]" \
                    experiment.name=GALE_CIFAR100_SWEEP logging.name="sweep_scaffold_gale_cc_${lr_decay}_${ridge_alpha}" &
  done
  wait
done
