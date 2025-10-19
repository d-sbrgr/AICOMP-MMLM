from dataclasses import dataclass


@dataclass
class CrossValidationConfig:
    n_splits: int = 5
    shuffle: bool = True
    seed: int = 42
