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
    logging.configs.wandb_entity=vaseline555 train.local_epochs=3 \
      optimizer.lr=0.1 optimizer.lr_decay.type=exponential optimizer.lr_decay.gamma=0.99 \
        experiment.name=GALE_NC logging.name="nc_cifar100_diri_fedavg_0.1_0.99" &

python -m appfl_sim.runner \
  --config appfl_sim/config/adaptive_local_steps/cifar100_diri/fedavg.yaml \
    logging.configs.wandb_entity=vaseline555 train.local_epochs=3 \
      optimizer.lr=0.1 optimizer.lr_decay.type=exponential optimizer.lr_decay.gamma=0.985 \
        experiment.name=GALE_NC logging.name="nc_cifar100_diri_fedavg_0.1_0.985" &

python -m appfl_sim.runner \
  --config appfl_sim/config/adaptive_local_steps/cifar100_diri/fedavg.yaml \
    logging.configs.wandb_entity=vaseline555 train.local_epochs=3 \
      optimizer.lr=0.1 optimizer.lr_decay.type=exponential optimizer.lr_decay.gamma=0.98 \
        experiment.name=GALE_NC logging.name="nc_cifar100_diri_fedavg_0.1_0.98" &

python -m appfl_sim.runner \
  --config appfl_sim/config/adaptive_local_steps/cifar100_diri/fedavg.yaml \
    logging.configs.wandb_entity=vaseline555 train.local_epochs=3 \
      optimizer.lr=0.1 optimizer.lr_decay.type=exponential optimizer.lr_decay.gamma=0.975 \
        experiment.name=GALE_NC logging.name="nc_cifar100_diri_fedavg_0.1_0.975" &
    
python -m appfl_sim.runner \
  --config appfl_sim/config/adaptive_local_steps/cifar100_diri/fedavg.yaml \
    logging.configs.wandb_entity=vaseline555 train.local_epochs=3 \
      optimizer.lr=0.1 optimizer.lr_decay.type=exponential optimizer.lr_decay.gamma=0.97 \
        experiment.name=GALE_NC logging.name="nc_cifar100_diri_fedavg_0.1_0.97"

wait
