from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVR

from ...dataloaders.base_dataloader import BaseDataloader
from ...evaluation import brier_score
from ...experiments import Tracker
from ...utils.constants import Columns, Metrics
from ..cross_validation.cv_config import CrossValidationConfig
from ..model import SupervisedModel
from .config import SVMHyperparamConfig


class SVMRegressorModel(SupervisedModel):
    """
    A wrapper class to train and validate Support Vector Machine models using scikit-learn.
    """

    def __init__(
        self,
        data: BaseDataloader,
        params: SVMHyperparamConfig,
        cv: CrossValidationConfig | None,
        tracker: Tracker,
    ) -> None:
        super().__init__(data, params, cv, tracker, StandardScaler())
        self.model: SVR | None = None

    def fit(self, season: int, start_season: int = 2003) -> None:
        """
        Fit the model on data provided by the given dataloader.

        The model is using data up to but not including the specified season.

        Args:
            season (int): Season for which the model should be validated.
            start_season (int): Start season for which the model should be trained on.
        """
        self.data.setup()
        X, y = self.data.train_data(season, start_season)
        X = self._drop_and_sort_features(X)

        self._fit_scaler(X)
        X = self._transform(X)

        self.model = SVR(**self.params.as_params())
        self.model.fit(X, y.values)

        preds = self.model.predict(X)
        preds = np.clip(preds, 0, 1)
        self.tracker.log({Metrics.TRAIN_BRIER: brier_score(y.values, preds)})

    def validate(self):
        """
        Validate the model for the season provided in `fit()`.

        Note:
            Fit must be called beforehand
        """
        assert self.model is not None, "Call fit() before validate()"
        X, y = self.data.valid_data()
        X = self._drop_and_sort_features(X)
        X = self._transform(X)  # Apply scaling
        preds = self.model.predict(X)
        preds = np.clip(preds, 0, 1)
        self.tracker.log({Metrics.VALID_BRIER: brier_score(y.values, preds)})

    def predict(self, matchups: pd.DataFrame) -> Iterable[float]:
        """
        Make predictions for the given matchups.

        Args:
            matchups (pd.DataFrame): Matchups to make predictions for.
        Note:
            Fit must be called first.
        """
        assert self.model is not None, "Call fit() before predict()"
        X = self.data.test_data(matchups)
        X = self._drop_and_sort_features(X)
        X = self._transform(X)  # Apply scaling
        preds = self.model.predict(X)
        preds = np.clip(preds, 0, 1)
        return pd.Series(preds, index=X.index)

    def _drop_and_sort_features(self, df: pd.DataFrame) -> pd.DataFrame:
        id_cols = [Columns.SEASON, Columns.T1_TEAM_ID, Columns.T2_TEAM_ID]
        df = df.drop(columns=[c for c in id_cols if c in df.columns])
        return df
