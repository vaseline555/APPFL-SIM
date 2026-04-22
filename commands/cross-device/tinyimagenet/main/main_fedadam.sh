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

# Run - E = 1
for i in "${!SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i python -m appfl_sim.runner \
    --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=fedadam \
      "logging.configs.wandb_tags=cross-device,tinyimagenet,fedadam,main" \
      train.local_epochs=1 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.001 \
      optimizer.lr_decay.gamma=0.999 \
      "experiment.seed=${SEEDS[$i]}" \
      experiment.name=GALE_TINYIMAGENET \
      logging.name="main_tinyimagenet_fedadam_1_${SEEDS[$i]}" &
done
wait

# Run - E = 2
for i in "${!SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i python -m appfl_sim.runner \
    --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=fedadam \
      "logging.configs.wandb_tags=cross-device,tinyimagenet,fedadam,main" \
      train.local_epochs=2 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.001 \
      optimizer.lr_decay.gamma=0.97 \
      "experiment.seed=${SEEDS[$i]}" \
      experiment.name=GALE_TINYIMAGENET \
      logging.name="main_tinyimagenet_fedadam_2_${SEEDS[$i]}" &
done
wait

# Run - E = 4
for i in "${!SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i python -m appfl_sim.runner \
    --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=fedadam \
      "logging.configs.wandb_tags=cross-device,tinyimagenet,fedadam,main" \
      train.local_epochs=4 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.001 \
      optimizer.lr_decay.gamma=0.96 \
      "experiment.seed=${SEEDS[$i]}" \
      experiment.name=GALE_TINYIMAGENET \
      logging.name="main_tinyimagenet_fedadam_4_${SEEDS[$i]}" &
done
wait

# Run - E = 8
for i in "${!SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i python -m appfl_sim.runner \
    --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=fedadam \
      "logging.configs.wandb_tags=cross-device,tinyimagenet,fedadam,main" \
      train.local_epochs=8 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.001 \
      optimizer.lr_decay.gamma=0.93 \
      "experiment.seed=${SEEDS[$i]}" \
      experiment.name=GALE_TINYIMAGENET \
      logging.name="main_tinyimagenet_fedadam_8_${SEEDS[$i]}" &
done
wait
