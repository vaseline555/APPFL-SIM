import pytest
from omegaconf import OmegaConf

from appfl_sim.misc.runtime_utils import _resolve_runtime_policies


def _dslinucb_c_config(*, num_clients: int, num_sampled_clients: int, holdout_clients: int = 0):
    eval_configs = {"scheme": "dataset"}
    if holdout_clients > 0:
        eval_configs = {
            "scheme": "client",
            "client_counts": int(holdout_clients),
        }

    return OmegaConf.create(
        {
            "algorithm": {
                "name": "gale_avg_c",
                "aggregator": "FedavgAggregator",
                "scheduler": "DslinucbCScheduler",
                "trainer": "FedavgTrainer",
            },
            "train": {
                "num_clients": int(num_clients),
                "num_sampled_clients": int(num_sampled_clients),
            },
            "eval": {
                "configs": eval_configs,
            },
        }
    )


def test_dslinucb_c_rejects_partial_participation():
    config = _dslinucb_c_config(num_clients=10, num_sampled_clients=5)

    with pytest.raises(ValueError, match="DslinucbCScheduler requires full participation"):
        _resolve_runtime_policies(config, {"num_clients": 10})


def test_dslinucb_c_allows_full_participation_among_train_clients():
    config = _dslinucb_c_config(
        num_clients=10,
        num_sampled_clients=8,
        holdout_clients=2,
    )

    policy = _resolve_runtime_policies(config, {"num_clients": 10})

    assert policy["num_sampled_clients"] == 8
    assert len(policy["train_client_ids"]) == 8
