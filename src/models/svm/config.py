from dataclasses import asdict, dataclass
from typing import Any

from ..hyperparam_config import HyperparamConfig


@dataclass
class SVMHyperparamConfig(HyperparamConfig):
    """
    Configuration for scikit-learn Support Vector Machine Regressor (SVR).
    Refer to https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    """

    kernel: str = "rbf"  # "linear", "poly", "rbf", "sigmoid", "precomputed"
    degree: int = 3  # Degree of polynomial kernel (ignored by other kernels)
    gamma: str | float = "scale"  # "scale", "auto", or float
    coef0: float = 0.0  # Independent term in kernel function (poly/sigmoid)
    tol: float = 1e-3  # Tolerance for stopping criterion
    C: float = 1.0  # Regularization parameter
    epsilon: float = 0.1  # Epsilon in epsilon-SVR model
    shrinking: bool = True  # Whether to use shrinking heuristic
    cache_size: float = 200  # Kernel cache size (in MB)
    verbose: bool = False
    max_iter: int = -1  # No limit (-1)

    def as_params(self) -> dict[str, Any]:
        """Convert config to scikit-learn parameters."""
        return asdict(self)

    def run_name(self) -> dict[str, Any]:
        """Return key hyperparameters for run naming."""
        return {
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "degree": self.degree if self.kernel == "poly" else None,
        }
