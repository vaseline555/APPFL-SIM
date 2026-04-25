#!/bin/bash -l
#PBS -N GALE
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=2:30:00
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

# Original loop/template
# # TinyImageNet Dirichlet Non-IID
# # for lr_decay in 0.999 0.995 0.99 0.985; do
# #   python -m appfl_sim.runner \
# #     --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml \
# #       train.local_epochs=1 \
# #         optimizer.lr=0.1 optimizer.lr_decay.gamma=$lr_decay \
# #           experiment.name=GALE_TINYIMAGENET_SWEEP \
# #             logging.name="sweep_fedavg_1_${lr_decay}" &
# # done
# # wait

# Flattened commands
CUDA_VISIBLE_DEVICES=0 python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml logging.backend=file train.local_epochs=1 optimizer.lr=0.01 optimizer.lr_decay.gamma=0.9975 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedavg_1_0.9999 &
CUDA_VISIBLE_DEVICES=1 python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml logging.backend=file train.local_epochs=1 optimizer.lr=0.1 optimizer.lr_decay.gamma=0.9975 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedavg_1_0.9995 &
CUDA_VISIBLE_DEVICES=2 python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml logging.backend=file train.local_epochs=1 optimizer.lr=0.0316 optimizer.lr_decay.gamma=0.9975 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedavg_1_0.9975 &
CUDA_VISIBLE_DEVICES=3 python -m appfl_sim.runner --config appfl_sim/config/cross-device/tinyimagenet/fedavg.yaml logging.backend=file train.local_epochs=1 optimizer.lr=0.0316 optimizer.lr_decay.gamma=0.9975 experiment.name=GALE_TINYIMAGENET_SWEEP logging.name=sweep_fedavg_1_0.995 &
