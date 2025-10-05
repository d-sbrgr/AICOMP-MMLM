import pandas as pd
import numpy as np

from typing import Iterable

from .model import Model


class RandomModel(Model):
    """A model that predicts the outcome of a matchup randomly with a 50% chance for each team."""

    def __init__(self):
        super().__init__()

    def fit(self, season: int) -> None:
        pass

    def predict(self, matchups: pd.DataFrame) -> Iterable[float]:
        return np.full(len(matchups), 0.5)
