from typing import Any

import wandb  # lazy import

from .base_tracker import Tracker
from .config import ExperimentConfig


class WandbTracker(Tracker):
    def __init__(self, exp: ExperimentConfig, hparams: dict[str, Any]):
        self.run = None
        self.exp = exp
        self.hparams = hparams
        self.run = None

    def start(self) -> None:
        """Initialize a new run to be logged to wandb."""
        self.run = wandb.init(
            project=self.exp.project,
            entity=self.exp.entity,
            name=self.exp.name,
            config=self.hparams,
            notes=self.exp.notes,
            tags=list(self.exp.tags),
            reinit=True,
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log the given metrics at the given step to wandb."""
        if self.run is not None:
            self.run.log(metrics, step=step)

    def config(self) -> dict[str, Any]:
        """Return the hyperparameters for the given run"""
        return self.hparams

    def finish(self) -> None:
        """Finish the current run."""
        if self.run is not None:
            self.run.finish()
            self.run = None
