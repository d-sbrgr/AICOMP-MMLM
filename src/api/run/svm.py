"""SVM model training wrapper - uses generic runner."""

import wandb

from ...models.svm import EnsembleSVMRegressorModel, SVMRegressorModel
from ...models.svm.config import SVMHyperparamConfig
from .generic import run_model, sweep_model


def sweep_svm(config=None, **kwargs):
    """
    Wrapper for W&B agent run for SVM models.

    Uses cross-validation by default.

    Args:
        config: W&B sweep configuration
        **kwargs: Additional arguments to pass to wandb.init()
    """
    sweep_model(
        SVMRegressorModel,
        SVMHyperparamConfig,
        "svm_config",
        cv_config=None,
        config=config,
        **kwargs,
    )


def run_svm(run: wandb.Run):
    """
    Train and validate an SVMRegressorModel using the given config with wandb experiment tracking.

    Uses cross-validation by default.

    The config is a dictionary defining the following fields:

        - run_config: Key-value mapping initializing a `RunConfig` object
        - svm_config: Key-value mapping initializing a `SVMHyperparamConfig` object
        - experiment_config: Key-value mapping initializing an `ExperimentConfig` object

    Note:
        The field "name" in the `ExperimentConfig` is filled automatically, do not define this in the config

    Args:
        run (wandb.Run): W&B experiment tracking run
    """
    run_model(run, SVMRegressorModel, SVMHyperparamConfig, "svm_config", cv_config=None)


def sweep_svm_ensemble(config=None, **kwargs):
    """
    Wrapper for W&B agent run for SVM models.

    Uses cross-validation by default.

    Args:
        config: W&B sweep configuration
        **kwargs: Additional arguments to pass to wandb.init()
    """
    sweep_model(
        EnsembleSVMRegressorModel,
        SVMHyperparamConfig,
        "svm_config",
        cv_config=None,
        config=config,
        **kwargs,
    )


def run_svm_ensemble(run: wandb.Run):
    """
    Train and validate an SVMRegressorModel using the given config with wandb experiment tracking.

    Uses cross-validation by default.

    The config is a dictionary defining the following fields:

        - run_config: Key-value mapping initializing a `RunConfig` object
        - svm_config: Key-value mapping initializing a `SVMHyperparamConfig` object
        - experiment_config: Key-value mapping initializing an `ExperimentConfig` object

    Note:
        The field "name" in the `ExperimentConfig` is filled automatically, do not define this in the config

    Args:
        run (wandb.Run): W&B experiment tracking run
    """
    run_model(run, EnsembleSVMRegressorModel, SVMHyperparamConfig, "svm_config", cv_config=None)
