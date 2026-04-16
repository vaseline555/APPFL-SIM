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

for lr_decay in 0.98 0.99 0.995; do
  for contexts in "[l]" "[d]" "[l,d]"; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/cross-silo/cifar100/gale_prox_c.yaml \
        logging.configs.wandb_entity=vaseline555 \
          optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
            algorithm.scheduler_kwargs.discount_gamma=0.80 \
              algorithm.scheduler_kwargs.exploration_alpha=0.0001 \
                algorithm.scheduler_kwargs.reward_scale=10 \
                  algorithm.trainer_kwargs.mu=0.3 \  
                    "algorithm.scheduler_kwargs.contexts=${contexts}" \
                      experiment.name=GALE_CIFAR100_001 logging.name="GALE_PROX_C_80_0.0001_${lr_decay}_${contexts}" &
  done
  wait
done
