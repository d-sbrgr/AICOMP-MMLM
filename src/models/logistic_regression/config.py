from dataclasses import asdict, dataclass
from typing import Any

from ..hyperparam_config import HyperparamConfig


@dataclass
class LogisticRegressionHyperparamConfig(HyperparamConfig):
    """
    Configuration for scikit-learn Logistic Regression.
    Refer to https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    penalty: str | None = "l2"  # "l1", "l2", "elasticnet", None
    dual: bool = False  # Dual formulation (only for l2 penalty with liblinear solver)
    tol: float = 1e-4  # Tolerance for stopping criteria
    C: float = 1.0  # Inverse of regularization strength (smaller = stronger regularization)
    fit_intercept: bool = True  # Add intercept/bias term
    intercept_scaling: float = 1.0  # Scaling for intercept (only with liblinear solver)
    class_weight: str | dict | None = None  # "balanced" or dict for class weights
    random_state: int = 42
    solver: str = "lbfgs"  # "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    max_iter: int = 1000  # Maximum number of iterations
    verbose: int = 0
    warm_start: bool = False  # Reuse solution from previous fit
    n_jobs: int = -1  # Use all available cores
    l1_ratio: float | None = None  # Elastic-Net mixing parameter (only for elasticnet penalty)

    def as_params(self) -> dict[str, Any]:
        """Convert config to scikit-learn parameters."""
        return asdict(self)

    def run_name(self) -> dict[str, Any]:
        """Return key hyperparameters for run naming."""
        return {
            "penalty": self.penalty,
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "l1_ratio": self.l1_ratio if self.penalty == "elasticnet" else None,
        }
