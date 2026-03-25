# CIFAR-10 Dirichlet Non-IID
for gamma in 0.67 0.80 0.90 0.95; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m appfl_sim.runner \
    --config appfl_sim/config/adaptive_local_steps/cifar10_diri/dsts.yaml \
      logging.configs.wandb_entity=vaseline555 experiment.seed=42 \
        algorithm.scheduler_kwargs.discount_gamma=$gamma \
          algorithm.scheduler_kwargs.likelihood_variance=1.0 \
            algorithm.scheduler_kwargs.prior_variance=1.0 \
              experiment.name=GenFL_Discounted2 \
                logging.name="cifar10_diri_dsts_cost_${gamma}_1_1" &

  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m appfl_sim.runner \
    --config appfl_sim/config/adaptive_local_steps/cifar10_diri/dsts.yaml \
      logging.configs.wandb_entity=vaseline555 experiment.seed=42 \
        algorithm.scheduler_kwargs.discount_gamma=$gamma \
          algorithm.scheduler_kwargs.likelihood_variance=0.1 \
            algorithm.scheduler_kwargs.prior_variance=1.0 \
              experiment.name=GenFL_Discounted2 \
                logging.name="cifar10_diri_dsts_cost_${gamma}_01_1" &
  
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m appfl_sim.runner \
    --config appfl_sim/config/adaptive_local_steps/cifar10_diri/dsts.yaml \
      logging.configs.wandb_entity=vaseline555 experiment.seed=42 \
        algorithm.scheduler_kwargs.discount_gamma=$gamma \
          algorithm.scheduler_kwargs.likelihood_variance=0.01 \
            algorithm.scheduler_kwargs.prior_variance=1.0 \
              experiment.name=GenFL_Discounted2 \
                logging.name="cifar10_diri_dsts_cost_${gamma}_001_1"
  wait

  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m appfl_sim.runner \
    --config appfl_sim/config/adaptive_local_steps/cifar10_diri/dsts.yaml \
      logging.configs.wandb_entity=vaseline555 experiment.seed=42 \
        algorithm.scheduler_kwargs.discount_gamma=$gamma \
          algorithm.scheduler_kwargs.likelihood_variance=0.1 \
            algorithm.scheduler_kwargs.prior_variance=0.1 \
              experiment.name=GenFL_Discounted2 \
                logging.name="cifar10_diri_dsts_cost_${gamma}_01_01" &

  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m appfl_sim.runner \
    --config appfl_sim/config/adaptive_local_steps/cifar10_diri/dsts.yaml \
      logging.configs.wandb_entity=vaseline555 experiment.seed=42 \
        algorithm.scheduler_kwargs.discount_gamma=$gamma \
          algorithm.scheduler_kwargs.likelihood_variance=0.01 \
            algorithm.scheduler_kwargs.prior_variance=0.1 \
              experiment.name=GenFL_Discounted2 \
                logging.name="cifar10_diri_dsts_cost_${gamma}_001_01" &
  
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m appfl_sim.runner \
    --config appfl_sim/config/adaptive_local_steps/cifar10_diri/dsts.yaml \
      logging.configs.wandb_entity=vaseline555 experiment.seed=42 \
        algorithm.scheduler_kwargs.discount_gamma=$gamma \
          algorithm.scheduler_kwargs.likelihood_variance=0.001 \
            algorithm.scheduler_kwargs.prior_variance=0.1 \
              experiment.name=GenFL_Discounted2 \
                logging.name="cifar10_diri_dsts_cost_${gamma}_0001_01"
  wait
done