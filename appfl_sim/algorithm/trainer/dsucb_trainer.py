from appfl_sim.algorithm.trainer.fedavg_trainer import FedavgTrainer


class DsucbTrainer(FedavgTrainer):
    @staticmethod
    def _attach_local_gen_error(result):
        if (
            isinstance(result, dict)
            and "pre_val_loss" in result
            and "pre_train_loss" in result
            and isinstance(result["pre_val_loss"], (int, float))
            and isinstance(result["pre_train_loss"], (int, float))
        ):
            result["local_gen_error"] = float(
                result["pre_val_loss"] - result["pre_train_loss"]
            )
        return result

    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        return self._attach_local_gen_error(result)
