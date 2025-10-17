from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    project: str = "xgboost-regressor"
    entity: str = "aicomp-mmlm"
    name: str = "xgboost-regressor"
    notes: str | None = None
    tags: tuple[str, ...] = ("xgb", "baseline")
    fast_dev_run: bool = False
