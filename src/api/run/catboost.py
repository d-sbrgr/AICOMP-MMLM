"""CatBoost model training wrapper - uses generic runner."""

import wandb

from ...models.catboost import CatBoostModel
from ...models.catboost.config import CatBoostHyperparamConfig
from .generic import run_model, sweep_model


def sweep_catboost(config=None, **kwargs):
    """
    Wrapper for W&B agent run for CatBoost models.

    Uses cross-validation by default.

    Args:
        config: W&B sweep configuration
        **kwargs: Additional arguments to pass to wandb.init()
    """
    sweep_model(
        CatBoostModel,
        CatBoostHyperparamConfig,
        "catboost_config",
        cv_config=None,
        config=config,
        **kwargs,
    )


def run_catboost(run: wandb.Run):
    """
    Train and validate a CatBoostModel using the given config with wandb experiment tracking.

    Uses cross-validation by default.

    The config is a dictionary defining the following fields:

        - run_config: Key-value mapping initializing a `RunConfig` object
        - catboost_config: Key-value mapping initializing a `CatBoostHyperparamConfig` object
        - experiment_config: Key-value mapping initializing an `ExperimentConfig` object

    Note:
        The field "name" in the `ExperimentConfig` is filled automatically, do not define this in the config

    Args:
        run (wandb.Run): W&B experiment tracking run
    """
    run_model(run, CatBoostModel, CatBoostHyperparamConfig, "catboost_config", cv_config=None)
