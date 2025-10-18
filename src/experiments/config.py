from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    project: str = "xgboost-regressor"
    entity: str = "aicomp-mmlm"
    name: str = "xgboost-regressor"
