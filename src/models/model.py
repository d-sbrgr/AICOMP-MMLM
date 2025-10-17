from collections.abc import Iterable

import pandas as pd


class Model:
    def __init__(self):
        pass

    def fit(self, season: int) -> None:
        """Fits the model to the data it requires.

        The training data may differ per model and hence needs to be loaded by each implementation.
        """
        raise NotImplementedError()

    def predict(self, matchups: pd.DataFrame) -> Iterable[float]:
        """Predicts the outcome of the given matchups.

        Args:
            matchups (pd.DataFrame): A dataframe containing the matchups to predict. The dataframe must contain
                the following columns:
                    - 'Season': The season of the matchup.
                    - 'TeamID1': The ID of the first team.
                    - 'TeamID2': The ID of the second team.
        Returns:
            pd.Series: A series containing the predicted outcome of the matchups. The index of the series must
                match the index of the input dataframe.
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.__class__.__name__.removesuffix("Model")
