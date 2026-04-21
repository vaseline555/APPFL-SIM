#!/bin/bash -l
#PBS -N GALE
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:30:00
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

# Run
for i in "${!SEEDS[@]}"; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/mnist/gale_avg_c.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=gale_avg_c \
      "logging.configs.wandb_tags=cross-silo,mnist,gale_avg_c,main" \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.01 \
      optimizer.lr_decay.gamma=0.98 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
      algorithm.scheduler_kwargs.exploration_beta=0.01 \
      algorithm.scheduler_kwargs.ridge_alpha=1.0 \
      algorithm.scheduler_kwargs.reward_scale=1 \
      algorithm.scheduler_kwargs.contexts="[l,d]" \
      "experiment.seed=${SEEDS[$i]}" \
      experiment.name=GALE_MNIST \
      logging.name="main_mnist_fedavg_gale_cc_${SEEDS[$i]}" &
done
wait
