from dataclasses import asdict, dataclass
from typing import Any

from ..hyperparam_config import HyperparamConfig


@dataclass
class CatBoostHyperparamConfig(HyperparamConfig):
    """
    Configuration for CatBoost Regressor.
    Refer to https://catboost.ai/en/docs/concepts/python-reference_catboostregressor

    Note: CatBoost is a tree-based model and doesn't require feature scaling.
    """

    # Training parameters
    iterations: int = 400  # Number of boosting iterations
    learning_rate: float = 0.03  # Step size shrinkage
    depth: int = 6  # Depth of the trees

    # Regularization
    l2_leaf_reg: float = 3.0  # L2 regularization coefficient
    random_strength: float = 1.0  # Amount of randomness for scoring splits
    bagging_temperature: float = 1.0  # Controls intensity of Bayesian bootstrap

    # Sampling
    subsample: float = 0.8  # Sample rate for bagging

    # Performance
    task_type: str = "CPU"  # "CPU" or "GPU"
    thread_count: int = -1  # Number of threads (-1 = all cores)

    # Tree structure
    border_count: int = 254  # Number of splits for numerical features
    grow_policy: str = "SymmetricTree"  # "SymmetricTree", "Lossguide", "Depthwise"
    min_data_in_leaf: int = 1  # Minimum number of training samples in a leaf

    # Other
    random_seed: int = 42
    verbose: int = 0  # 0 = silent, higher = more verbose
    allow_writing_files: bool = False  # Don't write snapshot files

    # Loss function
    loss_function: str = "RMSE"  # "RMSE", "MAE", "Quantile", etc.

    def as_params(self) -> dict[str, Any]:
        """Convert config to CatBoost parameters (excludes scaling params)."""
        params = asdict(self)
        params.pop("iterations", None)
        return params

    def run_name(self) -> dict[str, Any]:
        """Return key hyperparameters for run naming."""
        return {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "subsample": self.subsample,
            "grow_policy": self.grow_policy,
        }
