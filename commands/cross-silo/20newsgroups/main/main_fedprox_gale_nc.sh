#!/bin/bash -l

set -euo pipefail
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

# Run
for i in "${!SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/20newsgroups/gale_prox.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=gale_prox \
      "logging.configs.wandb_tags=cross-silo,20newsgroups,gale_prox,main" \
      optimizer.lr_decay.enable=true \
      optimizer.lr=5e-05 \
      optimizer.lr_decay.gamma=0.98 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
      algorithm.scheduler_kwargs.exploration_alpha=0.0005 \
      algorithm.scheduler_kwargs.reward_scale=1 \
      algorithm.trainer_kwargs.mu=0.3 \
      "experiment.seed=${SEEDS[$i]}" \
      experiment.name=GALE_20NEWSGROUPS \
      logging.name="main_20newsgroups_fedprox_gale_nc_${SEEDS[$i]}" &
done
wait
