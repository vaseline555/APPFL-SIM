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
# # Run - E = 1
# for i in "${!SEEDS[@]}"; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=scaffold \
#       "logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main" \
#       train.local_epochs=1 \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=5e-05 \
#       optimizer.lr_decay.gamma=0.999 \
#       "experiment.seed=${SEEDS[$i]}" \
#       experiment.name=GALE_20NEWSGROUPS \
#       logging.name="main_20newsgroups_scaffold_1_${SEEDS[$i]}" &
# done
# wait
#
# # Run - E = 2
# for i in "${!SEEDS[@]}"; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=scaffold \
#       "logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main" \
#       train.local_epochs=2 \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=5e-05 \
#       optimizer.lr_decay.gamma=0.97 \
#       "experiment.seed=${SEEDS[$i]}" \
#       experiment.name=GALE_20NEWSGROUPS \
#       logging.name="main_20newsgroups_scaffold_2_${SEEDS[$i]}" &
# done
# wait
#
# # Run - E = 4
# for i in "${!SEEDS[@]}"; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=scaffold \
#       "logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main" \
#       train.local_epochs=4 \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=5e-05 \
#       optimizer.lr_decay.gamma=0.96 \
#       "experiment.seed=${SEEDS[$i]}" \
#       experiment.name=GALE_20NEWSGROUPS \
#       logging.name="main_20newsgroups_scaffold_4_${SEEDS[$i]}" &
# done
# wait
#
# # Run - E = 8
# for i in "${!SEEDS[@]}"; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml \
#       "logging.configs.wandb_entity=${WANDB_ENTITY}" \
#       "logging.configs.wandb_mode=${WANDB_MODE}" \
#       logging.configs.wandb_group=scaffold \
#       "logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main" \
#       train.local_epochs=8 \
#       optimizer.lr_decay.enable=true \
#       optimizer.lr=5e-05 \
#       optimizer.lr_decay.gamma=0.925 \
#       "experiment.seed=${SEEDS[$i]}" \
#       experiment.name=GALE_20NEWSGROUPS \
#       logging.name="main_20newsgroups_scaffold_8_${SEEDS[$i]}" &
# done
# wait

# Flattened commands
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.999 experiment.seed=1115802892 experiment.name=GALE_20NEWSGROUPS logging.name=main_20newsgroups_scaffold_1_1115802892
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.999 experiment.seed=2998691888 experiment.name=GALE_20NEWSGROUPS logging.name=main_20newsgroups_scaffold_1_2998691888
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main train.local_epochs=1 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.999 experiment.seed=1991082795 experiment.name=GALE_20NEWSGROUPS logging.name=main_20newsgroups_scaffold_1_1991082795
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main train.local_epochs=2 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.97 experiment.seed=1115802892 experiment.name=GALE_20NEWSGROUPS logging.name=main_20newsgroups_scaffold_2_1115802892
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main train.local_epochs=2 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.97 experiment.seed=2998691888 experiment.name=GALE_20NEWSGROUPS logging.name=main_20newsgroups_scaffold_2_2998691888
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main train.local_epochs=2 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.97 experiment.seed=1991082795 experiment.name=GALE_20NEWSGROUPS logging.name=main_20newsgroups_scaffold_2_1991082795
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main train.local_epochs=4 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.96 experiment.seed=1115802892 experiment.name=GALE_20NEWSGROUPS logging.name=main_20newsgroups_scaffold_4_1115802892
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main train.local_epochs=4 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.96 experiment.seed=2998691888 experiment.name=GALE_20NEWSGROUPS logging.name=main_20newsgroups_scaffold_4_2998691888
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main train.local_epochs=4 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.96 experiment.seed=1991082795 experiment.name=GALE_20NEWSGROUPS logging.name=main_20newsgroups_scaffold_4_1991082795
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main train.local_epochs=8 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.925 experiment.seed=1115802892 experiment.name=GALE_20NEWSGROUPS logging.name=main_20newsgroups_scaffold_8_1115802892
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main train.local_epochs=8 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.925 experiment.seed=2998691888 experiment.name=GALE_20NEWSGROUPS logging.name=main_20newsgroups_scaffold_8_2998691888
CUDA_VISIBLE_DEVICES= python -m appfl_sim.runner --config appfl_sim/config/cross-silo/20newsgroups/scaffold.yaml logging.configs.wandb_entity=${WANDB_ENTITY} logging.configs.wandb_mode=${WANDB_MODE:-online} logging.configs.wandb_group=scaffold logging.configs.wandb_tags=cross-silo,20newsgroups,scaffold,main train.local_epochs=8 optimizer.lr_decay.enable=true optimizer.lr=5e-05 optimizer.lr_decay.gamma=0.925 experiment.seed=1991082795 experiment.name=GALE_20NEWSGROUPS logging.name=main_20newsgroups_scaffold_8_1991082795
