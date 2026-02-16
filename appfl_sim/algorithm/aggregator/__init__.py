from .base_aggregator import BaseAggregator

# Optional APPFL aggregators for broad compatibility.
_OPTIONAL = {
    "FedAvgAggregator": ".fedavg_aggregator",
    "FedAvgMAggregator": ".fedavgm_aggregator",
    "FedAdamAggregator": ".fedadam_aggregator",
    "FedYogiAggregator": ".fedyogi_aggregator",
    "FedAdagradAggregator": ".fedadagrad_aggregator",
    "FedAsyncAggregator": ".fedasync_aggregator",
    "FedBuffAggregator": ".fedbuff_aggregator",
    "FedCompassAggregator": ".fedcompass_aggregator",
    "IIADMMAggregator": ".iiadmm_aggregator",
    "ICEADMMAggregator": ".iceadmm_aggregator",
    "FedSBAggregator": ".fedsb_aggregator",
}

__all__ = ["BaseAggregator"]
for cls_name, mod_name in _OPTIONAL.items():
    try:
        module = __import__(f"appfl_sim.algorithm.aggregator{mod_name}", fromlist=[cls_name])
        globals()[cls_name] = getattr(module, cls_name)
        __all__.append(cls_name)
    except Exception:  # pragma: no cover
        continue
