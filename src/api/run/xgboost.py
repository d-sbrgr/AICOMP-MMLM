import wandb

from ...dataloaders.xgboost import SeasonAverageDataLoader, XGBDataLoader
from ...experiments import ExperimentConfig, WandbTracker
from ...models.xgboost import XGBRegressorModel
from ...models.xgboost.config import CrossValidationConfig, XGBHyperparamConfig, XGBRunConfig, get_run_name
from ...utils import unflatten_config


def sweep_xgboost(config=None, **kwargs):
    """Wrapper for W&B agent run (LLM inference mode)."""
    with wandb.init(config=config, **kwargs) as run:
        run_xgboost(run)


def run_xgboost(run: wandb.Run):
    """
    Train and validate an XGBRegressorModel using the given config with wandb experiment tracking.

    The config is a dictionary defining the following fields:

        - run_config: Key-value mapping initializing a `XGBRunConfig` object
        - xgboost_config: Key-value mapping initializing a `XGBHyperparamConfig` object
        - experiment_config: Key-value mapping initializing an `ExperimentConfig` object

    Note:
        The field "name" in the `ExperimentConfig` is filled automatically, do not define this in the config

    Args:
        run (wandb.Run): W&B experiment tracking run
    """
    config = unflatten_config(run.config)
    run_config = XGBRunConfig(**config.get("run_config", {}))

    data_loader: XGBDataLoader = {"season_average": SeasonAverageDataLoader}.get(
        run_config.data_loader, SeasonAverageDataLoader
    )(run_config.num_features)

    hyperparameters = XGBHyperparamConfig(**config.get("xgboost_config", {}))
    experiment_config = ExperimentConfig(
        **config.get("experiment_config", {}), name=get_run_name(hyperparameters, run_config)
    )

    # Initialize the experiment tracker and model
    tracker = WandbTracker(experiment_config, run)
    model = XGBRegressorModel(data_loader, hyperparameters, CrossValidationConfig(), tracker)

    # Train and validate the model using the given parameters
    model.fit(run_config.valid_season, run_config.start_season)
    model.validate()
