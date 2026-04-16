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

# TinyImageNet Dirichlet Non-IID
for lr_decay in 0.999 0.995 0.99 0.985; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-device/tinyimagenet/fedadam.yaml \
      train.local_epochs=1 \
        optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
          experiment.name=GALE_TINYIMAGENET_SWEEP \
            logging.name="sweep_fedadam_1_${lr_decay}" &
done
wait
