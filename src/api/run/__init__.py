__all__ = ["RUN_TARGET", "run_xgboost"]

from collections.abc import Callable
from typing import Any

from .xgboost import run_xgboost

RUN_TARGET = Callable[[dict[str, Any]], None]
