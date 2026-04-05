#!/bin/bash

#SBATCH -J GALE_FEDPROX
#SBATCH --output=GALE_FEDPROX_%j.out
#SBATCH --error=GALE_FEDPROX_%j.log
#SBATCH -A m5073 
#SBATCH -C gpu                    
#SBATCH -q regular                 
#SBATCH -N 1                   
#SBATCH -G 2                       
#SBATCH -t 2:30:00    

# Load environment
source .venv/bin/activate


# 260403 - E=5 gamma=0.1
# CIFAR-10 Dirichlet Non-IID
for mu in 0.0001 0.001 0.01 0.1; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/adaptive_local_steps/cifar100_diri/fedprox.yaml \
        logging.configs.wandb_entity=vaseline555 train.local_epochs=5 \
          optimizer.lr=0.01 optimizer.lr_decay.step_size=10 optimizer.lr_decay.gamma=0.1 \
            algorithm.trainer_kwargs.mu=$mu \
              experiment.name=GALE_NC_CIFAR100 logging.name="fedprox_5_${mu}" &
done
wait