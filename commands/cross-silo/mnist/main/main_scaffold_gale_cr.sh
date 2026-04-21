#!/bin/bash -l

set -euo pipefail
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# Original loop/template
# # Generate uncorrelated 32-bit seeds for each run
# mapfile -t SEEDS < <(
# python3 - <<'PY'
# from numpy.random import SeedSequence
# MASTER_SEED = 52525959
# NUM_RUNS = 3
#
# children = SeedSequence(MASTER_SEED).spawn(NUM_RUNS)
# for child in children:
#     print(int(child.generate_state(1, dtype="uint32")[0]))
# PY
# )
#
# # Run
# for i in "${!SEEDS[@]}"; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-silo/mnist/gale_scaffold_r.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=gale_scaffold_r \
#       "logging.configs.wandb_tags=cross-silo,mnist,gale_scaffold_r,main" \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=0.01 \
#       optimizer.lr_decay.gamma=0.98 \
#       algorithm.scheduler_kwargs.discount_gamma=0.80 \
#       algorithm.scheduler_kwargs.exploration_beta=0.01 \
#       algorithm.scheduler_kwargs.ridge_alpha=1.0 \
#       algorithm.scheduler_kwargs.reward_scale=1 \
#       algorithm.scheduler_kwargs.contexts="[l,d]" \
#       "experiment.seed=${SEEDS[$i]}" \
#       experiment.name=GALE_MNIST \
#       logging.name="main_mnist_scaffold_gale_cr_${SEEDS[$i]}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/mnist/gale_scaffold_r.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=gale_scaffold_r logging.configs.wandb_tags=cross-silo,mnist,gale_scaffold_r,main optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.98 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_beta=0.01 algorithm.scheduler_kwargs.ridge_alpha=1.0 algorithm.scheduler_kwargs.reward_scale=1 algorithm.scheduler_kwargs.contexts=[l,d] experiment.seed=1115802892 experiment.name=GALE_MNIST logging.name=main_mnist_scaffold_gale_cr_1115802892
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/mnist/gale_scaffold_r.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=gale_scaffold_r logging.configs.wandb_tags=cross-silo,mnist,gale_scaffold_r,main optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.98 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_beta=0.01 algorithm.scheduler_kwargs.ridge_alpha=1.0 algorithm.scheduler_kwargs.reward_scale=1 algorithm.scheduler_kwargs.contexts=[l,d] experiment.seed=2998691888 experiment.name=GALE_MNIST logging.name=main_mnist_scaffold_gale_cr_2998691888
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/mnist/gale_scaffold_r.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=gale_scaffold_r logging.configs.wandb_tags=cross-silo,mnist,gale_scaffold_r,main optimizer.lr_decay.enable=true optimizer.lr=0.01 optimizer.lr_decay.gamma=0.98 algorithm.scheduler_kwargs.discount_gamma=0.80 algorithm.scheduler_kwargs.exploration_beta=0.01 algorithm.scheduler_kwargs.ridge_alpha=1.0 algorithm.scheduler_kwargs.reward_scale=1 algorithm.scheduler_kwargs.contexts=[l,d] experiment.seed=1991082795 experiment.name=GALE_MNIST logging.name=main_mnist_scaffold_gale_cr_1991082795
