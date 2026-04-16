#!/bin/bash -l
#PBS -N GALE
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=5:30:00
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

# CIFAR-100 Dirichlet Non-IID
for lr_decay in 0.965 0.96 0.955 0.95; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/cifar100/fedavg.yaml \
      logging.configs.wandb_entity=vaseline555 train.local_epochs=8 \
        optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
          experiment.name=GALE_CIFAR100_SWEEP logging.name="sweep_fedavg_8_${lr_decay}" &
done
wait

