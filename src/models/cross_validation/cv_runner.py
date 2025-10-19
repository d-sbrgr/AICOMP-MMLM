"""
A utility class to make cross-validation folds during model training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .cv_config import CrossValidationConfig


class CVRunner:
    def __init__(self, cv_cfg: CrossValidationConfig):
        self.cv_cfg = cv_cfg

    def split(self, y: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
        skf = StratifiedKFold(n_splits=self.cv_cfg.n_splits, shuffle=self.cv_cfg.shuffle, random_state=self.cv_cfg.seed)
        idx = np.arange(len(y))
        return list(skf.split(idx, y.round().astype(int)))
