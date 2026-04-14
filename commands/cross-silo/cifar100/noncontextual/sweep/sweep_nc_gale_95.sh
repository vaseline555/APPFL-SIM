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

for alpha in 0.0001 0.001 0.01 0.1 10; do
  for gamma in 0.98 0.99 0.995; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/adaptive_local_steps/cifar100_diri/dsucb.yaml \
        logging.configs.wandb_entity=vaseline555 \
          optimizer.lr=0.001 optimizer.lr_decay.gamma=$gamma \
            algorithm.scheduler_kwargs.discount_gamma=0.95 \
              algorithm.scheduler_kwargs.exploration_alpha="$alpha" \
                algorithm.scheduler_kwargs.mul_factor=10 \
                  experiment.name=GALE_CIFAR100_001 logging.name="GALE_95_${alpha}_${gamma}" &
  done
  wait
done