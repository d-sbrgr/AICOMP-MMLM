"""Utility functions for creating submissions from W&B runs."""

from ..experiments.config import RunConfig
from .base_dataloader import BaseDataloader
from .ensemble.ensemble_season_avg_dataloader import EnsembleSeasonAverageDataLoader
from .simple.season_avg_dataloader import SeasonAverageDataLoader
from .simple.sliding_window_avg_dataloader import SlidingWindowAvgDataLoader
from .simple.weighted_season_avg_dataloader import WeightedSeasonAvgDataLoader


def get_data_loader(run_config: RunConfig) -> BaseDataloader:
    """
    Get the appropriate data loader based on run configuration.

    Args:
        run_config: The run configuration containing data loader settings

    Returns:
        An instance of the appropriate dataloader class
    """
    data_loader_map = {
        "season_average": SeasonAverageDataLoader,
        "season_average_ensemble": EnsembleSeasonAverageDataLoader,
        "weighted_season_average": WeightedSeasonAvgDataLoader,
        "sliding_window_average": SlidingWindowAvgDataLoader,
    }

    loader_cls = data_loader_map.get(run_config.data_loader, SeasonAverageDataLoader)

    data_loader_config = run_config.data_loader_config
    if data_loader_config is None:
        return loader_cls(run_config.num_features)
    return loader_cls(run_config.num_features, **data_loader_config)
