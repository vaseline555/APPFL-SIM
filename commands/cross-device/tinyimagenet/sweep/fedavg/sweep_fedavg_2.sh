#!/bin/bash -l
#PBS -N GALE
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=4:30:00
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
# for lr_decay in 0.995 0.99 0.985 0.98; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
#       train.local_epochs=2 \
#         optimizer.lr=0.1 optimizer.lr_decay.gamma=$lr_decay \
#           experiment.name=GALE_TINYIMAGENET_SWEEP \
#             logging.name="sweep_fedavg_2_${lr_decay}" &
# done
# wait

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
    train.local_epochs=2 \
      optimizer.lr=0.1 optimizer.lr_decay.gamma=0.995 \
        experiment.name=GALE_TINYIMAGENET_SWEEP \
          logging.name="sweep_fedavg_2_0.995"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
    train.local_epochs=2 \
      optimizer.lr=0.1 optimizer.lr_decay.gamma=0.99 \
        experiment.name=GALE_TINYIMAGENET_SWEEP \
          logging.name="sweep_fedavg_2_0.99"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
    train.local_epochs=2 \
      optimizer.lr=0.1 optimizer.lr_decay.gamma=0.985 \
        experiment.name=GALE_TINYIMAGENET_SWEEP \
          logging.name="sweep_fedavg_2_0.985"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
    train.local_epochs=2 \
      optimizer.lr=0.1 optimizer.lr_decay.gamma=0.98 \
        experiment.name=GALE_TINYIMAGENET_SWEEP \
          logging.name="sweep_fedavg_2_0.98"
