from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.svm import SVR

from ...dataloaders.base_dataloader import BaseDataloader
from ...evaluation import brier_score
from ...experiments import Tracker
from ...utils.constants import Columns, Metrics
from ..cross_validation.cv_config import CrossValidationConfig
from ..cross_validation.cv_runner import CVRunner
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
        super().__init__(data, params, cv, tracker)
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

        self._do_cross_validation(X, y)

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
        preds = self.model.predict(X)
        preds = np.clip(preds, 0, 1)
        return pd.Series(preds, index=X.index)

    def _drop_and_sort_features(self, df: pd.DataFrame) -> pd.DataFrame:
        id_cols = [Columns.SEASON, Columns.T1_TEAM_ID, Columns.T2_TEAM_ID]
        df = df.drop(columns=[c for c in id_cols if c in df.columns])
        return df

    def _do_cross_validation(self, X, y):
        if self.cv_cfg is None:
            return

        cv_runner = CVRunner(self.cv_cfg)
        splits = cv_runner.split(y)
        out_of_frame = np.zeros(len(y))

        for fold, (tr_idx, va_idx) in enumerate(splits):
            svm_model = SVR(**self.params.as_params())
            svm_model.fit(X.iloc[tr_idx], y.values[tr_idx])

            preds = svm_model.predict(X.iloc[va_idx])
            preds = np.clip(preds, 0, 1)
            out_of_frame[va_idx] = preds

            fold_score = brier_score(y.values[va_idx], preds)
            self.tracker.log({f"{Metrics.CV_BRIER}/fold_{fold}": fold_score})

        cv_score = brier_score(y.values, out_of_frame)
        self.tracker.log({Metrics.CV_BRIER: cv_score})
