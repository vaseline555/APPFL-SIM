#!/bin/bash -l

# Generate uncorrelated 32-bit seeds for each run
mapfile -t SEEDS < <(
python3 - <<'PY'
from numpy.random import SeedSequence
MASTER_SEED = 555
NUM_RUNS = 3

children = SeedSequence(MASTER_SEED).spawn(NUM_RUNS)
for child in children:
    print(int(child.generate_state(1, dtype="uint32")[0]))
PY
)

# Run - E = 1
for i in "${!SEEDS[@]}"; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/cross-silo/cifar100/scaffold.yaml \
        logging.configs.wandb_entity=vaseline555 train.local_epochs=1 \
          optimizer.lr=0.001 optimizer.lr_decay.gamma=0.999 \
            "experiment.seed=${SEEDS[$i]}" \
              experiment.name=GALE_CIFAR100 logging.name="main_cifar100_scaffold_1_${SEEDS[$i]}" &
done
wait

# Run - E = 2
for i in "${!SEEDS[@]}"; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/cross-silo/cifar100/scaffold.yaml \
        logging.configs.wandb_entity=vaseline555 train.local_epochs=2 \
         optimizer.lr=0.001 optimizer.lr_decay.gamma=0.97 \
            "experiment.seed=${SEEDS[$i]}" \
              experiment.name=GALE_CIFAR100 logging.name="main_cifar100_scaffold_2_${SEEDS[$i]}" &
done
wait

# Run - E = 4
for i in "${!SEEDS[@]}"; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/cross-silo/cifar100/scaffold.yaml \
        logging.configs.wandb_entity=vaseline555 train.local_epochs=4 \
          optimizer.lr=0.001 optimizer.lr_decay.gamma=0.96 \
            "experiment.seed=${SEEDS[$i]}" \
              experiment.name=GALE_CIFAR100 logging.name="main_cifar100_scaffold_4_${SEEDS[$i]}" &
done
wait

# Run - E = 8
for i in "${!SEEDS[@]}"; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/cross-silo/cifar100/scaffold.yaml \
        logging.configs.wandb_entity=vaseline555 train.local_epochs=8 \
          optimizer.lr=0.001 optimizer.lr_decay.gamma=0.925 \
            "experiment.seed=${SEEDS[$i]}" \
              experiment.name=GALE_CIFAR100 logging.name="main_cifar100_scaffold_8_${SEEDS[$i]}" &
done
wait