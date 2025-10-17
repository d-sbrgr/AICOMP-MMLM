"""
An experiment tracking logging to stdout.
"""

from typing import Any

from .base_tracker import Tracker


class DefaultTracker(Tracker):
    def __init__(self, cfg: dict[str, Any]):
        self._cfg = cfg

    def start(self) -> None:
        """No effect for the default tracker."""
        pass

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Print the given metrics to stdout."""
        print(f"metrics: {metrics}, step: {step}")

    def config(self) -> dict[str, Any]:
        """Return the tracker config."""
        return self._cfg

    def finish(self) -> None:
        """No effect for the default tracker."""
        pass
