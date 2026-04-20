#!/bin/bash -l
#PBS -N GALE
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=2:00:00
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

# londonsmartmeter sweep for gale_adam
for lr_decay in 0.93 0.95 0.97 0.99; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-device/londonsmartmeter/gale_adam.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=gale_adam \
      "logging.configs.wandb_tags=cross-device,londonsmartmeter,gale_adam,sweep" \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.001 \
      optimizer.lr_decay.gamma=$lr_decay \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
      algorithm.scheduler_kwargs.exploration_alpha=0.001 \
      algorithm.scheduler_kwargs.reward_scale=1 \
      experiment.name=GALE_LONDONSMARTMETER_SWEEP \
      logging.name="sweep_londonsmartmeter_gale_adam_80_0.001_${lr_decay}" &
done
wait
