"""XGBoost model training wrapper - uses generic runner."""

import wandb

from ...models.cross_validation.cv_config import CrossValidationConfig
from ...models.xgboost import XGBRegressorModel
from ...models.xgboost.config import XGBHyperparamConfig
from .generic import run_model, sweep_model


def sweep_xgboost(config=None, **kwargs):
    """
    Wrapper for W&B agent run for XGBoost models.

    Uses cross-validation by default.

    Args:
        config: W&B sweep configuration
        **kwargs: Additional arguments to pass to wandb.init()
    """
    sweep_model(
        XGBRegressorModel,
        XGBHyperparamConfig,
        "xgboost_config",
        cv_config=CrossValidationConfig(),
        config=config,
        **kwargs,
    )


def run_xgboost(run: wandb.Run):
    """
    Train and validate an XGBRegressorModel using the given config with wandb experiment tracking.

    Uses cross-validation by default.

    The config is a dictionary defining the following fields:

        - run_config: Key-value mapping initializing a `RunConfig` object
        - xgboost_config: Key-value mapping initializing a `XGBHyperparamConfig` object
        - experiment_config: Key-value mapping initializing an `ExperimentConfig` object

    Note:
        The field "name" in the `ExperimentConfig` is filled automatically, do not define this in the config

    Args:
        run (wandb.Run): W&B experiment tracking run
    """
    run_model(run, XGBRegressorModel, XGBHyperparamConfig, "xgboost_config", cv_config=CrossValidationConfig())
