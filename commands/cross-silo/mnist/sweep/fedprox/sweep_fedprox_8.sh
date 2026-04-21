#!/bin/bash -l

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Original loop/template
# mnist sweep for fedprox with E=8
for mu in 0.001 0.01 0.1 1.0; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/mnist/fedprox.yaml \
      logging.backend=file \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=fedprox \
      "logging.configs.wandb_tags=cross-silo,mnist,fedprox,sweep" \
      train.local_epochs=8 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.01 \
      optimizer.lr_decay.gamma=0.965 \
      algorithm.trainer_kwargs.mu=$mu \
      experiment.name=GALE_MNIST_SWEEP \
      logging.name="sweep_mnist_fedprox_8_${mu}" &
done
wait
