import pytest
import torch
from omegaconf import OmegaConf

from appfl_sim.algorithm.aggregator import FedadamAggregator, FedavgAggregator
from appfl_sim.misc.config_utils import _resolve_algorithm_components


class _TinyStateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer("running", torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer("counter", torch.tensor([0], dtype=torch.int64))


def _state(weight: float, running: float, counter: int):
    return {
        "weight": torch.tensor([weight], dtype=torch.float32),
        "running": torch.tensor([running], dtype=torch.float32),
        "counter": torch.tensor([counter], dtype=torch.int64),
    }


def test_fedavg_sample_ratio_uses_current_round_only_for_partial_participation():
    aggregator = FedavgAggregator(
        model=_TinyStateModel(),
        aggregator_configs=OmegaConf.create(
            {"client_weights_mode": "sample_ratio", "optimize_memory": False}
        ),
    )

    aggregator.aggregate(
        {
            0: _state(weight=100.0, running=1000.0, counter=100),
            2: _state(weight=300.0, running=3000.0, counter=300),
        },
        sample_sizes={0: 10, 2: 30},
    )

    result = aggregator.aggregate(
        {
            0: _state(weight=2.0, running=10.0, counter=2),
            1: _state(weight=6.0, running=30.0, counter=6),
        },
        sample_sizes={0: 1, 1: 3},
    )

    assert torch.allclose(result["weight"], torch.tensor([5.0]))
    assert torch.allclose(result["running"], torch.tensor([25.0]))
    assert torch.equal(result["counter"], torch.tensor([5], dtype=torch.int64))


def test_fedadam_uses_current_round_weights_for_parameters_and_buffers():
    aggregator = FedadamAggregator(
        model=_TinyStateModel(),
        aggregator_configs=OmegaConf.create(
            {
                "client_weights_mode": "sample_ratio",
                "optimize_memory": False,
                "server_learning_rate": 1.0,
                "beta1": 0.0,
                "beta2": 1.0,
                "tau": 1.0,
            }
        ),
    )

    aggregator.aggregate(
        {
            0: _state(weight=100.0, running=1000.0, counter=100),
            2: _state(weight=300.0, running=3000.0, counter=300),
        },
        sample_sizes={0: 10, 2: 30},
    )

    result = aggregator.aggregate(
        {
            0: _state(weight=2.0, running=10.0, counter=2),
            1: _state(weight=6.0, running=30.0, counter=6),
        },
        sample_sizes={0: 1, 1: 3},
    )

    assert torch.allclose(result["weight"], torch.tensor([5.0]))
    assert torch.allclose(result["running"], torch.tensor([25.0]))
    assert torch.equal(result["counter"], torch.tensor([5], dtype=torch.int64))


def test_adaptive_client_weight_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported client_weights_mode"):
        FedavgAggregator(
            model=_TinyStateModel(),
            aggregator_configs=OmegaConf.create(
                {"client_weights_mode": "adaptive", "optimize_memory": False}
            ),
        )


def test_fednova_algorithm_name_is_no_longer_resolved():
    config = OmegaConf.create({"algorithm": {"name": "fednova"}})

    with pytest.raises(ValueError, match="FednovaAggregator"):
        _resolve_algorithm_components(config)
