"""Neural Network model training wrapper - uses generic runner."""

import wandb

from ...models.neural_network import NeuralNetworkModel
from ...models.neural_network.config import NeuralNetworkHyperparamConfig
from .generic import run_model, sweep_model


def sweep_neural_network(config=None, **kwargs):
    """
    Wrapper for W&B agent run for Neural Network models.

    Uses cross-validation by default.

    Args:
        config: W&B sweep configuration
        **kwargs: Additional arguments to pass to wandb.init()
    """
    sweep_model(
        NeuralNetworkModel,
        NeuralNetworkHyperparamConfig,
        "neural_network_config",
        cv_config=None,
        config=config,
        **kwargs,
    )


def run_neural_network(run: wandb.Run):
    """
    Train and validate a NeuralNetworkModel using the given config with wandb experiment tracking.

    Uses cross-validation by default.

    The config is a dictionary defining the following fields:

        - run_config: Key-value mapping initializing a `RunConfig` object
        - neural_network_config: Key-value mapping initializing a `NeuralNetworkHyperparamConfig` object
        - experiment_config: Key-value mapping initializing an `ExperimentConfig` object

    Note:
        The field "name" in the `ExperimentConfig` is filled automatically, do not define this in the config

    Args:
        run (wandb.Run): W&B experiment tracking run
    """
    run_model(run, NeuralNetworkModel, NeuralNetworkHyperparamConfig, "neural_network_config", cv_config=None)
