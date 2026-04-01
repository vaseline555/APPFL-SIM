#!/bin/bash -l
#PBS -k doe
#PBS -N GALE_TS
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=8:00:00
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
for lv in 0.0001 0.001 0.01 0.01 1.0; do
  for pv in 0.001 0.01 0.1 1.0 10.0; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/adaptive_local_steps/cifar100_diri/dsts.yaml \
        logging.configs.wandb_entity=vaseline555 \
            algorithm.scheduler_kwargs.discount_gamma=0.9 \
              algorithm.scheduler_kwargs.likelihood_variance=$lv \
                algorithm.scheduler_kwargs.prior_variance=$pv \
                  experiment.name=GALE_NC \
                    logging.name="nc_cifar100_diri_dsts_0.9_${lv}_${pv}" &
  done
  wait
done