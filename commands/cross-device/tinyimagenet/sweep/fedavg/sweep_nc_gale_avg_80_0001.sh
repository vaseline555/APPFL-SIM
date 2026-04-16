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

for lr_decay in 0.98 0.99 0.995; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-device/tinyimagenet/gale_avg.yaml \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_alpha=0.0001 \
            algorithm.scheduler_kwargs.reward_scale=10 \
              experiment.name=GALE_TINYIMAGENET \
                logging.name="sweep_gale_avg_80_0.0001_${lr_decay}" &
done
wait
