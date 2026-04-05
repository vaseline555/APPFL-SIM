#!/bin/bash

#SBATCH -J GALE_FEDAVG
#SBATCH --output=GALE_FEDAVG_%j.out
#SBATCH --error=GALE_FEDAVG_%j.log
#SBATCH -A m5073 
#SBATCH -C gpu                    
#SBATCH -q regular                 
#SBATCH -N 1                   
#SBATCH -G 2                       
#SBATCH -t 2:00:00    

# Load environment
source .venv/bin/activate

# 260403 - E=4 gamma=0.1
# CIFAR-10 Dirichlet Non-IID
for lr in 0.0005623 0.0003162 0.00017782 0.0001; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/adaptive_local_steps/cifar100_diri/fedavg.yaml \
      logging.configs.wandb_entity=vaseline555 train.local_epochs=4 \
        optimizer.lr=$lr optimizer.lr_decay.milestones="50,75" optimizer.lr_decay.gamma=0.5 \
          experiment.name=GALE_NC_CIFAR100 logging.name="fedavg_4_${lr}" &
done
wait



