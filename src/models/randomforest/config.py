from dataclasses import asdict, dataclass
from typing import Any

from ..hyperparam_config import HyperparamConfig


@dataclass
class RandomForestHyperparamConfig(HyperparamConfig):
    """
    Configuration for scikit-learn Random Forest Regressor.
    Refer to https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    """

    n_estimators: int = 400
    max_depth: int | None = 6
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | float = "sqrt"  # "sqrt", "log2", or float for fraction
    bootstrap: bool = True
    max_samples: float | None = 0.8  # subsample ratio when bootstrap=True
    min_impurity_decrease: float = 0.0
    max_leaf_nodes: int | None = None
    min_weight_fraction_leaf: float = 0.0
    random_state: int = 42
    n_jobs: int = -1  # use all available cores
    verbose: int = 0
    warm_start: bool = False
    ccp_alpha: float = 0.0  # complexity parameter for pruning

    def as_params(self) -> dict[str, Any]:
        """Convert config to scikit-learn parameters."""
        return asdict(self)

    def run_name(self) -> dict[str, Any]:
        """Return key hyperparameters for run naming."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "max_samples": self.max_samples,
            "ccp_alpha": self.ccp_alpha,
        }
