"""Generic model training and validation runner for W&B experiments."""

import wandb

from ...dataloaders.base_dataloader import BaseDataloader
from ...dataloaders.simple import FeatureSelectionDataLoader, SeasonAverageDataLoader
from ...experiments import ExperimentConfig, Tracker, WandbTracker
from ...experiments.config import RunConfig, get_run_name
from ...models.cross_validation.cv_config import CrossValidationConfig
from ...models.hyperparam_config import HyperparamConfig
from ...models.model import SupervisedModel
from ...utils import unflatten_config


def sweep_model(
    model_cls: type[SupervisedModel],
    hyperparameter_cls: type[HyperparamConfig],
    config_key: str = "model_config",
    cv_config: CrossValidationConfig | None = None,
    config=None,
    **kwargs,
):
    """
    Wrapper for W&B agent run.

    Args:
        model_cls: The model class to instantiate (e.g., XGBRegressorModel, RandomForestRegressorModel)
        hyperparameter_cls: The hyperparameter config class (e.g., XGBHyperparamConfig, RandomForestHyperparamConfig)
        config_key: The key in the W&B config for the model hyperparameters (default: "model_config")
        cv_config: Cross-validation configuration. If None, no cross-validation is performed.
        config: W&B sweep configuration
        **kwargs: Additional arguments to pass to wandb.init()
    """
    with wandb.init(config=config, **kwargs) as run:
        run_model(run, model_cls, hyperparameter_cls, config_key, cv_config)


def run_model(
    run: wandb.Run,
    model_cls: type[SupervisedModel],
    hyperparameter_cls: type[HyperparamConfig],
    config_key: str = "model_config",
    cv_config: CrossValidationConfig | None = None,
):
    """
    Train and validate a model using the given config with wandb experiment tracking.

    The config is a dictionary defining the following fields:

        - run_config: Key-value mapping initializing a `RunConfig` object
        - {config_key}: Key-value mapping initializing the hyperparameter config object
        - experiment_config: Key-value mapping initializing an `ExperimentConfig` object

    Note:
        The field "name" in the `ExperimentConfig` is filled automatically, do not define this in the config

    Args:
        run (wandb.Run): W&B experiment tracking run
        model_cls: The model class to instantiate (e.g., XGBRegressorModel, RandomForestRegressorModel)
        hyperparameter_cls: The hyperparameter config class (e.g., XGBHyperparamConfig, RandomForestHyperparamConfig)
        config_key: The key in the W&B config for the model hyperparameters (default: "model_config")
        cv_config: Cross-validation configuration. If None, no cross-validation is performed.
    """
    config = unflatten_config(run.config)
    run_config = RunConfig(**config.get("run_config", {}))

    data_loader: BaseDataloader = _get_data_loader(run_config)

    hyperparameters = hyperparameter_cls(**config.get(config_key, {}))

    experiment_config = ExperimentConfig(
        **config.get("experiment_config", {}), name=get_run_name(hyperparameters, run_config)
    )

    tracker: Tracker = WandbTracker(experiment_config, run)

    model = model_cls(data_loader, hyperparameters, cv_config, tracker)
    model.fit(run_config.valid_season, run_config.start_season)
    model.validate()


def _get_data_loader(run_config: RunConfig) -> BaseDataloader:
    """Get the appropriate data loader based on run configuration."""
    data_loader_map = {
        "season_average": SeasonAverageDataLoader,
        "feature_selection": FeatureSelectionDataLoader,
    }

    loader_cls = data_loader_map.get(run_config.data_loader, SeasonAverageDataLoader)
    return loader_cls(run_config.num_features)
