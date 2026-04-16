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
  for context_tag in l d ld; do
    case "$context_tag" in
      l) contexts='[l]' ;;
      d) contexts='[d]' ;;
      ld) contexts='[l,d]' ;;
    esac

    python -m appfl_sim.runner \
      --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
        optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
          algorithm.scheduler_kwargs.discount_gamma=0.80 \
            algorithm.scheduler_kwargs.exploration_beta=0.001 \
              algorithm.scheduler_kwargs.reward_scale=10 \
                "algorithm.scheduler_kwargs.contexts=${contexts}" \
                  experiment.name=GALE_TINYIMAGENET_SWEEP \
                    logging.name="sweep_gale_adam_r_80_0.001_${lr_decay}_${context_tag}" &
  done
  wait
done
