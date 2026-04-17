#!/bin/bash -l
#PBS -N GALE
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=7:30:00
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
# for lr_decay in 0.965 0.96 0.955 0.95; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
#       train.local_epochs=8 \
#         optimizer.lr=0.1 optimizer.lr_decay.gamma=$lr_decay \
#           experiment.name=GALE_TINYIMAGENET_SWEEP \
#             logging.name="sweep_fedavg_8_${lr_decay}" &
# done
# wait

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
    train.local_epochs=8 \
      optimizer.lr=0.1 optimizer.lr_decay.gamma=0.965 \
        experiment.name=GALE_TINYIMAGENET_SWEEP \
          logging.name="sweep_fedavg_8_0.965"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
    train.local_epochs=8 \
      optimizer.lr=0.1 optimizer.lr_decay.gamma=0.96 \
        experiment.name=GALE_TINYIMAGENET_SWEEP \
          logging.name="sweep_fedavg_8_0.96"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
    train.local_epochs=8 \
      optimizer.lr=0.1 optimizer.lr_decay.gamma=0.955 \
        experiment.name=GALE_TINYIMAGENET_SWEEP \
          logging.name="sweep_fedavg_8_0.955"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
    train.local_epochs=8 \
      optimizer.lr=0.1 optimizer.lr_decay.gamma=0.95 \
        experiment.name=GALE_TINYIMAGENET_SWEEP \
          logging.name="sweep_fedavg_8_0.95"
