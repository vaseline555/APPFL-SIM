from appfl_sim.algorithm.trainer.fedavg_trainer import FedavgTrainer


class DsucbTrainer(FedavgTrainer):
    @staticmethod
    def _attach_local_gen_error(result, *, enabled):
        return FedavgTrainer._attach_local_gen_error(result, enabled=enabled)

    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        return self._attach_local_gen_error(
            result,
            enabled=bool(self.train_configs.get("track_gen_rewards", False)),
        )
