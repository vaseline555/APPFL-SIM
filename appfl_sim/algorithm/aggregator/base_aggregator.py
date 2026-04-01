import abc
from typing import Any, Dict, Union, OrderedDict, Tuple


class BaseAggregator:
    def __init__(self) -> None:
        self.client_sample_size: Dict[Union[str, int], int] = {}

    def set_client_sample_size(self, client_id: Union[str, int], sample_size: int):
        """Set the sample size of a client"""
        self.client_sample_size[client_id] = sample_size

    @abc.abstractmethod
    def aggregate(
        self, *args, **kwargs
    ) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Aggregate local model(s) from clients and return the global model
        """
        pass

    @abc.abstractmethod
    def get_parameters(
        self, **kwargs
    ) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """Return global model parameters"""
        pass

    def get_round_metrics(
        self,
        *,
        client_train_stats: Dict[Union[str, int], Dict[str, Any]],
        sample_sizes: Dict[Union[str, int], int],
    ) -> Dict[str, Any]:
        del client_train_stats, sample_sizes
        return {}
