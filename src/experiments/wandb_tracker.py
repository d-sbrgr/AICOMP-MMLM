from typing import Any

import wandb  # lazy import

from .base_tracker import Tracker
from .config import ExperimentConfig


class WandbTracker(Tracker):
    def __init__(self, exp: ExperimentConfig, run: wandb.Run):
        self.run = run
        self.run.name = exp.name
        self.hparams = run.config

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log the given metrics at the given step to wandb."""
        self.run.log(metrics, step=step)

    def config(self) -> dict[str, Any]:
        """Return the hyperparameters for the given run"""
        return self.hparams
