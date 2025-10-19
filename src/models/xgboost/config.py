from dataclasses import asdict, dataclass
from typing import Any

from ..hyperparam_config import HyperparamConfig


@dataclass
class XGBHyperparamConfig(HyperparamConfig):
    """
    Refer to https://xgboost.readthedocs.io/en/stable/parameter.html
    """

    num_rounds: int = 400
    device: str = "cpu"  # or "cuda"
    objective: str = "reg:squarederror"
    booster: str = "gbtree"
    learning_rate: float = 0.01
    gamma: float = 0.0
    max_depth: int = 6
    min_child_weight: float = 1.0
    max_delta_step: int = 0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    colsample_bylevel: float = 1.0
    colsample_bynode: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    tree_method: str = "hist"
    max_bin: int = 256
    grow_policy: str = "lossguide"
    seed: int = 42

    def as_params(self) -> dict[str, Any]:
        params = asdict(self)
        del params["num_rounds"]
        return params

    def run_name(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "booster": self.booster,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
        }
