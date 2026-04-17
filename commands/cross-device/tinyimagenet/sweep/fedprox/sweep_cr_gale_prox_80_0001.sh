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

# for lr_decay in 0.98 0.99 0.995; do
#   for contexts in "[l]" "[d]" "[l,d]"; do
#     for ridge_alpha in 0.1 1.0 10.0; do
#       python -m appfl_sim.runner \
#         --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
#           logging.configs.wandb_entity=vaseline555 \
#             optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
#               algorithm.scheduler_kwargs.discount_gamma=0.80 \
#                 algorithm.scheduler_kwargs.exploration_beta=0.0001 \
#                   algorithm.scheduler_kwargs.ridge_alpha=$ridge_alpha \
#                   algorithm.scheduler_kwargs.reward_scale=1 \
#                     "algorithm.scheduler_kwargs.contexts=${contexts}" \
#                       algorithm.trainer_kwargs.mu=0.3 \
#                         experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_${lr_decay}_${contexts}_${ridge_alpha}" &
#     done
#   done
#   wait
# done

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=0.1 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.98_[l]_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=1.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.98_[l]_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=10.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.98_[l]_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=0.1 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.98_[d]_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=1.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.98_[d]_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=10.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.98_[d]_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=0.1 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l,d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.98_[l,d]_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=1.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l,d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.98_[l,d]_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=10.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l,d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.98_[l,d]_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=0.1 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.99_[l]_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=1.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.99_[l]_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=10.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.99_[l]_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=0.1 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.99_[d]_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=1.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.99_[d]_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=10.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.99_[d]_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=0.1 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l,d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.99_[l,d]_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=1.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l,d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.99_[l,d]_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=10.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l,d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.99_[l,d]_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=0.1 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.995_[l]_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=1.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.995_[l]_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=10.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.995_[l]_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=0.1 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.995_[d]_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=1.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.995_[d]_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=10.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.995_[d]_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=0.1 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l,d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.995_[l,d]_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=1.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l,d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.995_[l,d]_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_prox_r.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.0001 \
            algorithm.scheduler_kwargs.ridge_alpha=10.0 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              "algorithm.scheduler_kwargs.contexts=[l,d]" \
                algorithm.trainer_kwargs.mu=0.3 \
                  experiment.name=GALE_TINYIMAGENET_SWEEP logging.name="GALE_PROX_R_80_0.0001_0.995_[l,d]_10.0"
