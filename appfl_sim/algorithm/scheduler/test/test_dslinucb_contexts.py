from omegaconf import OmegaConf

from appfl_sim.algorithm.scheduler.dslinucb_c_scheduler import DslinucbCScheduler
from appfl_sim.algorithm.scheduler.dslinucb_r_scheduler import DslinucbRScheduler


def _scheduler_config(*, contexts):
    return OmegaConf.create(
        {
            "num_clients": 4,
            "action_space": [1, 2],
            "contexts": contexts,
            "base_lr": 0.25,
            "lr_decay": {
                "enable": False,
                "type": "none",
            },
        }
    )


def test_dslinucb_c_supports_single_context_subject():
    scheduler = DslinucbCScheduler(
        scheduler_configs=_scheduler_config(contexts="t"),
        aggregator=object(),
        logger=None,
    )
    scheduler._latest_client_pre_train_losses = {
        1: 0.3,
        3: 0.75,
    }

    kwargs = scheduler.get_pull_kwargs(selected_ids=[1, 3], round_idx=5)

    assert scheduler.context_subjects == ["t"]
    assert scheduler.context_dim == 1
    assert kwargs["client_contexts"] == {
        1: [0.3],
        3: [0.75],
    }


def test_dslinucb_r_supports_multiple_context_subjects_from_list():
    scheduler = DslinucbRScheduler(
        scheduler_configs=_scheduler_config(contexts=["l", "t", "v"]),
        aggregator=object(),
        logger=None,
    )
    scheduler._latest_client_pre_train_losses = {
        0: 1.5,
        2: 4.0,
    }
    scheduler._latest_client_pre_val_losses = {
        0: 0.5,
        2: 0.25,
    }
    scheduler._latest_client_context_weights = {
        0: 2.0,
        2: 5.0,
    }

    kwargs = scheduler.get_pull_kwargs(selected_ids=[0, 2], round_idx=3)

    assert scheduler.context_subjects == ["l", "t", "v"]
    assert scheduler.context_dim == 3
    assert kwargs["client_contexts"] == [
        [0.25, 1.5, 0.5],
        [0.25, 4.0, 0.25],
    ]
    assert kwargs["client_weights"] == [2.0, 5.0]


def test_dslinucb_c_round_metrics_include_assigned_local_steps():
    scheduler = DslinucbCScheduler(
        scheduler_configs=_scheduler_config(contexts="l"),
        aggregator=object(),
        logger=None,
    )

    metrics = scheduler.get_round_metrics(round_local_steps={3: 8, 1: 2})

    assert metrics["policy"] == {
        "tau_t_clients": 2,
        "tau_t_mean": 5.0,
        "tau_t_min": 2,
        "tau_t_max": 8,
    }
    assert metrics["assigned_local_steps"] == {
        "1": 2,
        "3": 8,
    }
