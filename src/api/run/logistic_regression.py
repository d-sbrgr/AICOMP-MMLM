"""Logistic Regression model training wrapper - uses generic runner."""

import wandb

from ...models.logistic_regression import LogisticRegressionModel
from ...models.logistic_regression.config import LogisticRegressionHyperparamConfig
from .generic import run_model, sweep_model


def sweep_logistic_regression(config=None, **kwargs):
    """
    Wrapper for W&B agent run for Logistic Regression models.

    Uses cross-validation by default.

    Args:
        config: W&B sweep configuration
        **kwargs: Additional arguments to pass to wandb.init()
    """
    sweep_model(
        LogisticRegressionModel,
        LogisticRegressionHyperparamConfig,
        "logistic_regression_config",
        cv_config=None,
        config=config,
        **kwargs,
    )


def run_logistic_regression(run: wandb.Run):
    """
    Train and validate a LogisticRegressionModel using the given config with wandb experiment tracking.

    Uses cross-validation by default.

    The config is a dictionary defining the following fields:

        - run_config: Key-value mapping initializing a `RunConfig` object
        - logistic_regression_config: Key-value mapping initializing a `LogisticRegressionHyperparamConfig` object
        - experiment_config: Key-value mapping initializing an `ExperimentConfig` object

    Note:
        The field "name" in the `ExperimentConfig` is filled automatically, do not define this in the config

    Args:
        run (wandb.Run): W&B experiment tracking run
    """
    run_model(
        run, LogisticRegressionModel, LogisticRegressionHyperparamConfig, "logistic_regression_config", cv_config=None
    )
