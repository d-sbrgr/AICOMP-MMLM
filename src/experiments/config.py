import time
from dataclasses import asdict, dataclass

from ..models.hyperparam_config import HyperparamConfig


@dataclass
class ExperimentConfig:
    project: str = "xgboost-regressor"
    entity: str = "aicomp-mmlm"
    name: str = "xgboost-regressor"


@dataclass
class RunConfig:
    num_features: int = 10
    valid_season: int = 2024
    start_season: int = 2003
    data_loader: str = "season_average"
    data_loader_config: dict = None


def get_run_name(hyperparameters: HyperparamConfig, run_config: RunConfig) -> str:
    """Return a meaningful run name for wandb experiment tracking."""
    items = asdict(run_config)
    items.update(hyperparameters.run_name())
    class_name = hyperparameters.__class__.__name__.lower()
    return f"{class_name}_{'_'.join(f'{k}-{v}' for k, v in sorted(items.items()))}_{time.strftime('%y%m%d-%H%M%S')}"
