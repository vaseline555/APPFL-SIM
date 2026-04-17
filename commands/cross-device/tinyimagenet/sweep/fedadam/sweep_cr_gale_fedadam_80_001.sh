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
#   for context_tag in l d ld; do
#     case "$context_tag" in
#       l) contexts='[l]' ;;
#       d) contexts='[d]' ;;
#       ld) contexts='[l,d]' ;;
#     esac
#
#     for ridge_alpha in 0.1 1.0 10.0; do
#       python -m appfl_sim.runner \
#         --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
#           optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
#             algorithm.scheduler_kwargs.discount_gamma=0.80 \
#               algorithm.scheduler_kwargs.exploration_beta=0.001 \
#                 algorithm.scheduler_kwargs.ridge_alpha=$ridge_alpha \
#                 algorithm.scheduler_kwargs.reward_scale=1 \
#                   "algorithm.scheduler_kwargs.contexts=${contexts}" \
#                     experiment.name=GALE_TINYIMAGENET_SWEEP \
#                       logging.name="sweep_gale_adam_r_80_0.001_${lr_decay}_${context_tag}_${ridge_alpha}" &
#     done
#   done
#   wait
# done

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=0.1 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.98_l_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=1.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.98_l_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=10.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.98_l_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=0.1 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.98_d_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=1.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.98_d_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=10.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.98_d_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=0.1 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l,d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.98_ld_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=1.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l,d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.98_ld_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=10.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l,d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.98_ld_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=0.1 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.99_l_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=1.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.99_l_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=10.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.99_l_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=0.1 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.99_d_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=1.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.99_d_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=10.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.99_d_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=0.1 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l,d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.99_ld_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=1.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l,d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.99_ld_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=10.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l,d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.99_ld_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=0.1 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.995_l_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=1.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.995_l_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=10.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.995_l_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=0.1 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.995_d_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=1.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.995_d_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=10.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.995_d_10.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=0.1 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l,d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.995_ld_0.1"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=1.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l,d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.995_ld_1.0"

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-device/tinyimagenet/gale_adam_r.yaml \
    optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
      algorithm.scheduler_kwargs.discount_gamma=0.80 \
        algorithm.scheduler_kwargs.exploration_beta=0.001 \
          algorithm.scheduler_kwargs.ridge_alpha=10.0 \
          algorithm.scheduler_kwargs.reward_scale=1 \
            "algorithm.scheduler_kwargs.contexts=[l,d]" \
              experiment.name=GALE_TINYIMAGENET_SWEEP \
                logging.name="sweep_gale_adam_r_80_0.001_0.995_ld_10.0"
