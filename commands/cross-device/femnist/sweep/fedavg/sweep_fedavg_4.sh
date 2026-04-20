#!/bin/bash -l
#PBS -N GALE
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=5:00:00
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

# femnist sweep for fedavg with E=4
for lr_decay in 0.98 0.97 0.96 0.95; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-device/femnist/fedavg.yaml \
      "logging.configs.wandb_entity=${WANDB_ENTITY}" \
      "logging.configs.wandb_mode=${WANDB_MODE}" \
      logging.configs.wandb_group=fedavg \
      "logging.configs.wandb_tags=cross-device,femnist,fedavg,sweep" \
      train.local_epochs=4 \
      optimizer.lr_decay.enable=true \
      optimizer.lr=0.001 \
      optimizer.lr_decay.gamma=$lr_decay \
      experiment.name=GALE_FEMNIST_SWEEP \
      logging.name="sweep_femnist_fedavg_4_${lr_decay}" &
done
wait
