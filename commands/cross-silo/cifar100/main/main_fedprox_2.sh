#!/bin/bash -l

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Generate uncorrelated 32-bit seeds for each run
mapfile -t SEEDS < <(
python3 - <<'PY'
from numpy.random import SeedSequence
MASTER_SEED = 52525959
NUM_RUNS = 3

children = SeedSequence(MASTER_SEED).spawn(NUM_RUNS)
for child in children:
    print(int(child.generate_state(1, dtype="uint32")[0]))
PY
)

# Run - E = 2
for i in "${!SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/cifar100/fedprox.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=fedprox \
      "logging.configs.wandb_tags=cross-silo,cifar100,fedprox,main" \
      train.local_epochs=2 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.001 \
      optimizer.lr_decay.gamma=0.985 \
      algorithm.trainer_kwargs.mu=0.1 \
      "experiment.seed=${SEEDS[$i]}" \
      experiment.name=GALE \
      logging.name="main_cifar100_fedprox_2_${SEEDS[$i]}" &
done
wait
