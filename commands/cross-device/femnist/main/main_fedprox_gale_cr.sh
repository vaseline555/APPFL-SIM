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
    --config appfl_sim/config/cross-device/femnist/gale_prox_r.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=gale_prox_r \
      "logging.configs.wandb_tags=cross-device,femnist,gale_prox_r,main" \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.01 \
      optimizer.lr_decay.gamma=0.98 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
      algorithm.scheduler_kwargs.exploration_beta=0.001 \
      algorithm.scheduler_kwargs.ridge_alpha=1.0 \
      algorithm.scheduler_kwargs.reward_scale=1 \
      "algorithm.scheduler_kwargs.contexts=[l,d]" \
      algorithm.trainer_kwargs.mu=0.3 \
      "experiment.seed=${SEEDS[$i]}" \
      experiment.name=GALE_FEMNIST \
      logging.name="main_femnist_gale_prox_r_${SEEDS[$i]}" &
done
wait
