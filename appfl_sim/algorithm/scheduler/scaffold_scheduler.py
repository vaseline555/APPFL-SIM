from appfl_sim.algorithm.scheduler.fedavg_scheduler import FedavgScheduler


class ScaffoldScheduler(FedavgScheduler):
    """Compatibility alias for FedAvg scheduling; SCAFFOLD state now lives in the aggregator."""
