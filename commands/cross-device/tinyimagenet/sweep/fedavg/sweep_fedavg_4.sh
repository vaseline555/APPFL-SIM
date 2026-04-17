#!/bin/bash -l
#PBS -N GALE
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=5:00:00
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
# for lr_decay in 0.98 0.97 0.96 0.95; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
#       train.local_epochs=4 \
#         optimizer.lr=0.1 optimizer.lr_decay.gamma=$lr_decay \
#           experiment.name=GALE_TINYIMAGENET_SWEEP \
#             logging.name="sweep_fedavg_4_${lr_decay}" &
# done
# wait

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
    train.local_epochs=4 \
      optimizer.lr=0.1 optimizer.lr_decay.gamma=0.98 \
        experiment.name=GALE_TINYIMAGENET_SWEEP \
          logging.name="sweep_fedavg_4_0.98"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
    train.local_epochs=4 \
      optimizer.lr=0.1 optimizer.lr_decay.gamma=0.97 \
        experiment.name=GALE_TINYIMAGENET_SWEEP \
          logging.name="sweep_fedavg_4_0.97"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
    train.local_epochs=4 \
      optimizer.lr=0.1 optimizer.lr_decay.gamma=0.96 \
        experiment.name=GALE_TINYIMAGENET_SWEEP \
          logging.name="sweep_fedavg_4_0.96"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
    train.local_epochs=4 \
      optimizer.lr=0.1 optimizer.lr_decay.gamma=0.95 \
        experiment.name=GALE_TINYIMAGENET_SWEEP \
          logging.name="sweep_fedavg_4_0.95"
