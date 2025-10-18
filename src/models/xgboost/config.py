import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class XGBHyperparamConfig:
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


@dataclass
class XGBRunConfig:
    num_features: int = 10
    valid_season: int = 2024
    start_season: int = 2003
    data_loader: str = "season_average"


def get_run_name(hyperparameters: XGBHyperparamConfig, run_config: XGBRunConfig) -> str:
    """Return a meaningful run name for wandb experiment tracking."""
    items = asdict(run_config)
    items.update(hyperparameters.run_name())
    return f"xgb_{'_'.join(f'{k}-{v}' for k, v in sorted(items.items()))}_{time.strftime('%y%m%d-%H%M%S')}"


@dataclass
class CrossValidationConfig:
    n_splits: int = 5
    shuffle: bool = True
    seed: int = 42
