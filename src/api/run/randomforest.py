import wandb

from ...dataloaders.simple import SeasonAverageDataLoader
from ...experiments import ExperimentConfig, WandbTracker
from ...experiments.config import RunConfig, get_run_name
from ...models.randomforest import RandomForestRegressorModel
from ...models.randomforest.config import RandomForestHyperparamConfig


def sweep_randomforest(config=None, **kwargs):
    """Wrapper for W&B agent run."""
    with wandb.init(config=config, **kwargs) as run:
        run_randomforest(run)


def run_randomforest(run: wandb.Run):
    """
    Train and validate a RandomForestRegressorModel using the given config with wandb experiment tracking.

    The config is a dictionary defining the following fields:

        - run_config: Key-value mapping initializing a `RunConfig` object
        - randomforest_config: Key-value mapping initializing a `RandomForestHyperparamConfig` object
        - experiment_config: Key-value mapping initializing an `ExperimentConfig` object

    Note:
        The field "name" in the `ExperimentConfig` is filled automatically, do not define this in the config

    Args:
        run (wandb.Run): W&B experiment tracking run
    """
    config = run.config
    run_config = RunConfig(**config.get("run_config", {}))

    data_loader = {"season_average": SeasonAverageDataLoader}.get(run_config.data_loader, SeasonAverageDataLoader)(
        run_config.num_features
    )

    hyperparameters = RandomForestHyperparamConfig(**config.get("randomforest_config", {}))
    experiment_config = ExperimentConfig(
        **config.get("experiment_config", {}), name=get_run_name(hyperparameters, run_config)
    )

    tracker = WandbTracker(experiment_config, run)
    model = RandomForestRegressorModel(data_loader, hyperparameters, None, tracker)

    model.fit(run_config.valid_season, run_config.start_season)
    model.validate()
