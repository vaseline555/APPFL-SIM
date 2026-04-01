#!/bin/bash -l
#PBS -k doe
#PBS -N GALE_NC_FEDAVG
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
python -m appfl_sim.runner \
  --config appfl_sim/config/adaptive_local_steps/cifar100_diri/fedavg.yaml \
    logging.configs.wandb_entity=vaseline555 train.local_epochs=1 \
      experiment.name=GALE_NC logging.name="nc_cifar100_diri_fedavg_1" &&

python -m appfl_sim.runner \
  --config appfl_sim/config/adaptive_local_steps/cifar100_diri/fedavg.yaml \
    logging.configs.wandb_entity=vaseline555 train.local_epochs=2 \
      experiment.name=GALE_NC logging.name="nc_cifar100_diri_fedavg_2" &&

python -m appfl_sim.runner \
  --config appfl_sim/config/adaptive_local_steps/cifar100_diri/fedavg.yaml \
    logging.configs.wandb_entity=vaseline555 train.local_epochs=3 \
      experiment.name=GALE_NC logging.name="nc_cifar100_diri_fedavg_3" &&

python -m appfl_sim.runner \
  --config appfl_sim/config/adaptive_local_steps/cifar100_diri/fedavg.yaml \
    logging.configs.wandb_entity=vaseline555 train.local_epochs=4 \
      experiment.name=GALE_NC logging.name="nc_cifar100_diri_fedavg_4" &&

python -m appfl_sim.runner \
  --config appfl_sim/config/adaptive_local_steps/cifar100_diri/fedavg.yaml \
    logging.configs.wandb_entity=vaseline555 train.local_epochs=5 \
      experiment.name=GALE_NC logging.name="nc_cifar100_diri_fedavg_5"

wait
