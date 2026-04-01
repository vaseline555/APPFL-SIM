#!/bin/bash -l
#PBS -k doe
#PBS -N GALE_UCB
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=4:00:00
#PBS -l filesystems=home:eagle
#PBS -m abe
#PBS -j eo

set -euo pipefail

cd "${PBS_O_WORKDIR}"

module use /soft/modulefiles
module load conda
conda activate base

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1



# CIFAR-10 Dirichlet Non-IID
for alpha in 0.00003 0.0001 0.0003 0.001 0.003; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/adaptive_local_steps/cifar100_diri/dsucb.yaml \
      logging.configs.wandb_entity=vaseline555 \
          algorithm.scheduler_kwargs.discount_gamma=0.99 \
            algorithm.scheduler_kwargs.exploration_alpha=$alpha \
              experiment.name=GALE_NC \
                logging.name="nc_cifar100_diri_dsucb_0.99_${alpha}" &
done
wait
   