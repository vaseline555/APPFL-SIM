#!/bin/bash -l

#PBS -N GALE
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=6:00:00
#PBS -l filesystems=home:eagle
#PBS -r y
#PBS -k doe
#PBS -j oe

set -euo pipefail

cd "${PBS_O_WORKDIR:-$PWD}"

module use /soft/modulefiles
module load conda
conda activate base

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"

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
    --config appfl_sim/config/cross-device/tinyimagenet/fedprox.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=fedprox \
      "logging.configs.wandb_tags=cross-device,tinyimagenet,fedprox,main" \
      train.local_epochs=1 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.1 \
      optimizer.lr_decay.gamma=0.999 \
      algorithm.trainer_kwargs.mu=0.01 \
      "experiment.seed=${SEEDS[$i]}" \
      experiment.name=GALE \
      logging.name="main_tinyimagenet_fedprox_1_${SEEDS[$i]}" &
done
wait

# Run - E = 2
for i in "${!SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i python -m appfl_sim.runner \
    --config appfl_sim/config/cross-device/tinyimagenet/fedprox.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=fedprox \
      "logging.configs.wandb_tags=cross-device,tinyimagenet,fedprox,main" \
      train.local_epochs=2 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.1 \
      optimizer.lr_decay.gamma=0.9975 \
      algorithm.trainer_kwargs.mu=0.01 \
      "experiment.seed=${SEEDS[$i]}" \
      experiment.name=GALE \
      logging.name="main_tinyimagenet_fedprox_2_${SEEDS[$i]}" &
done
wait

# Run - E = 4
for i in "${!SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i python -m appfl_sim.runner \
    --config appfl_sim/config/cross-device/tinyimagenet/fedprox.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=fedprox \
      "logging.configs.wandb_tags=cross-device,tinyimagenet,fedprox,main" \
      train.local_epochs=4 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.1 \
      optimizer.lr_decay.gamma=0.995 \
      algorithm.trainer_kwargs.mu=0.01 \
      "experiment.seed=${SEEDS[$i]}" \
      experiment.name=GALE \
      logging.name="main_tinyimagenet_fedprox_4_${SEEDS[$i]}" &
done
wait
