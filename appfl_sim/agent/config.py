from dataclasses import dataclass, field

from omegaconf import DictConfig, OmegaConf


def _default_client_train_configs() -> DictConfig:
    return OmegaConf.create(
        {
            "trainer": "VanillaTrainer",
            "device": "cpu",
            "mode": "epoch",
            "num_local_epochs": 1,
            "batch_size": 32,
            "eval_batch_size": 128,
            "num_workers": 0,
            "train_data_shuffle": True,
            "train_pin_memory": False,
            "eval_pin_memory": False,
            "dataloader_persistent_workers": False,
            "dataloader_prefetch_factor": 2,
            "optim": "SGD",
            "lr": 0.01,
            "weight_decay": 0.0,
            "max_grad_norm": 0.0,
            "client_logging_enabled": True,
            "do_pre_evaluation": True,
            "do_post_evaluation": True,
            "eval_metrics": ["acc1"],
        }
    )


def _default_server_configs() -> DictConfig:
    return OmegaConf.create(
        {
            "num_clients": 1,
            "num_global_epochs": 1,
            "num_sampled_clients": 1,
            "device": "cpu",
            "num_workers": 0,
            "eval_batch_size": 128,
            "eval_show_progress": True,
            "eval_metrics": ["acc1"],
        }
    )


@dataclass
class ServerAgentConfig:
    client_configs: DictConfig = field(
        default_factory=lambda: OmegaConf.create(
            {
                "train_configs": _default_client_train_configs(),
                "model_configs": OmegaConf.create({}),
            }
        )
    )
    server_configs: DictConfig = field(default_factory=_default_server_configs)


@dataclass
class ClientAgentConfig:
    train_configs: DictConfig = field(default_factory=_default_client_train_configs)
    model_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    data_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    comm_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    additional_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
