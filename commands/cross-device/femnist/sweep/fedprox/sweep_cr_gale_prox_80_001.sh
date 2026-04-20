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

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# femnist contextual sweep for gale_prox_r
for lr_decay in 0.98 0.99 0.995; do
  for context_tag in l d ld; do
    case "$context_tag" in
      l) contexts='[l]' ;;
      d) contexts='[d]' ;;
      ld) contexts='[l,d]' ;;
    esac
    for ridge_alpha in 0.1 1.0 10.0; do
      python -m appfl_sim.runner \
        --config appfl_sim/config/cross-device/femnist/gale_prox_r.yaml \
          "logging.configs.wandb_entity=${WANDB_ENTITY}" \
          "logging.configs.wandb_mode=${WANDB_MODE}" \
          logging.configs.wandb_group=gale_prox_r \
          "logging.configs.wandb_tags=cross-device,femnist,gale_prox_r,sweep" \
          optimizer.lr_decay.enable=true \
          optimizer.lr=0.01 \
          optimizer.lr_decay.gamma=$lr_decay \
          algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=$ridge_alpha \
          algorithm.scheduler_kwargs.reward_scale=1 \
          "algorithm.scheduler_kwargs.contexts=${contexts}" \
          algorithm.trainer_kwargs.mu=0.3 \
          experiment.name=GALE_FEMNIST_SWEEP \
          logging.name="sweep_femnist_gale_prox_r_80_0.01_${lr_decay}_${context_tag}_${ridge_alpha}" &
    done
  done
  wait
done
