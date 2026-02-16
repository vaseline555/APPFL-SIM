from dataclasses import dataclass, field

from omegaconf import DictConfig, OmegaConf


@dataclass
class ServerAgentConfig:
    client_configs: DictConfig = field(
        default_factory=lambda: OmegaConf.create(
            {
                "train_configs": OmegaConf.create({}),
                "model_configs": OmegaConf.create({}),
            }
        )
    )
    server_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))


@dataclass
class ClientAgentConfig:
    train_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    model_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    data_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    comm_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    additional_configs: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
