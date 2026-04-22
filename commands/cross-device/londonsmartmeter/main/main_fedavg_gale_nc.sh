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
    --config appfl_sim/config/cross-device/londonsmartmeter/gale_avg.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=gale_avg \
      "logging.configs.wandb_tags=cross-device,londonsmartmeter,gale_avg,main" \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.001 \
      optimizer.lr_decay.gamma=0.98 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
      algorithm.scheduler_kwargs.exploration_alpha=0.001 \
      algorithm.scheduler_kwargs.reward_scale=1 \
      "experiment.seed=${SEEDS[$i]}" \
      experiment.name=GALE_LONDONSMARTMETER \
      logging.name="main_londonsmartmeter_gale_avg_${SEEDS[$i]}" &
done
wait
