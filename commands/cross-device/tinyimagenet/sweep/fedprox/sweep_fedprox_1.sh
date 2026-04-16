#!/bin/bash -l
#PBS -N GALE
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
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
for mu in 0.05 0.07 0.1 0.2 0.3; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/cross-silo/cifar100/fedprox.yaml \
        logging.configs.wandb_entity=vaseline555 train.local_epochs=1 \
          optimizer.lr=0.001 optimizer.lr_decay.gamma=0.999 \
            algorithm.trainer_kwargs.mu=$mu \
              experiment.name=GALE_CIFAR100_001 logging.name="fedprox_1_${mu}" &
done
wait
