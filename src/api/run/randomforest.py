"""Random Forest model training wrapper - uses generic runner."""

import wandb

from ...models.randomforest import EnsembleRandomForestRegressorModel, RandomForestRegressorModel
from ...models.randomforest.config import RandomForestHyperparamConfig
from .generic import run_model, sweep_model


def sweep_randomforest(config=None, **kwargs):
    """
    Wrapper for W&B agent run for Random Forest models.

    Does not use cross-validation by default (cv_config=None).

    Args:
        config: W&B sweep configuration
        **kwargs: Additional arguments to pass to wandb.init()
    """
    sweep_model(
        RandomForestRegressorModel,
        RandomForestHyperparamConfig,
        "randomforest_config",
        cv_config=None,
        config=config,
        **kwargs,
    )


def run_randomforest(run: wandb.Run):
    """
    Train and validate a RandomForestRegressorModel using the given config with wandb experiment tracking.

    Does not use cross-validation by default (cv_config=None).

    The config is a dictionary defining the following fields:

        - run_config: Key-value mapping initializing a `RunConfig` object
        - randomforest_config: Key-value mapping initializing a `RandomForestHyperparamConfig` object
        - experiment_config: Key-value mapping initializing an `ExperimentConfig` object

    Note:
        The field "name" in the `ExperimentConfig` is filled automatically, do not define this in the config

    Args:
        run (wandb.Run): W&B experiment tracking run
    """
    run_model(run, RandomForestRegressorModel, RandomForestHyperparamConfig, "randomforest_config", cv_config=None)


def sweep_randomforest_ensemble(config=None, **kwargs):
    """
    Wrapper for W&B agent run for Random Forest models.

    Does not use cross-validation by default (cv_config=None).

    Args:
        config: W&B sweep configuration
        **kwargs: Additional arguments to pass to wandb.init()
    """
    sweep_model(
        EnsembleRandomForestRegressorModel,
        RandomForestHyperparamConfig,
        "randomforest_config",
        cv_config=None,
        config=config,
        **kwargs,
    )


def run_randomforest_ensemble(run: wandb.Run):
    """
    Train and validate a RandomForestRegressorModel using the given config with wandb experiment tracking.

    Does not use cross-validation by default (cv_config=None).

    The config is a dictionary defining the following fields:

        - run_config: Key-value mapping initializing a `RunConfig` object
        - randomforest_config: Key-value mapping initializing a `RandomForestHyperparamConfig` object
        - experiment_config: Key-value mapping initializing an `ExperimentConfig` object

    Note:
        The field "name" in the `ExperimentConfig` is filled automatically, do not define this in the config

    Args:
        run (wandb.Run): W&B experiment tracking run
    """
    run_model(
        run, EnsembleRandomForestRegressorModel, RandomForestHyperparamConfig, "randomforest_config", cv_config=None
    )
