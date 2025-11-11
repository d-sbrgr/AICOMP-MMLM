from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression

from ...dataloaders import BaseDataloader, EnsembleDataloader
from ...evaluation import brier_score
from ...experiments import Tracker
from ...utils.constants import Columns, Metrics
from ..cross_validation.cv_config import CrossValidationConfig
from ..model import SupervisedModel
from .config import LogisticRegressionHyperparamConfig


class LogisticRegressionModel(SupervisedModel):
    """
    A wrapper class to train and validate Logistic Regression models using scikit-learn.

    Note: While LogisticRegression is typically a classifier, we use it here for probability
    prediction in a regression context (predicting win probability between 0 and 1).
    """

    def __init__(
        self,
        data: BaseDataloader,
        params: LogisticRegressionHyperparamConfig,
        cv: CrossValidationConfig | None,
        tracker: Tracker,
    ) -> None:
        super().__init__(data, params, cv, tracker, StandardScaler())
        self.model: LogisticRegression | None = None

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

        self.model = LogisticRegression(**self.params.as_params())

        y_binary = (y.values >= 0.5).astype(int)
        self.model.fit(X, y_binary)

        preds = self.model.predict_proba(X)[:, 1]
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

        preds = self.model.predict_proba(X)[:, 1]
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

        preds = self.model.predict_proba(X)[:, 1]
        preds = np.clip(preds, 0, 1)
        return pd.Series(preds, index=X.index)

    def _drop_and_sort_features(self, df: pd.DataFrame) -> pd.DataFrame:
        id_cols = [Columns.SEASON, Columns.T1_TEAM_ID, Columns.T2_TEAM_ID]
        df = df.drop(columns=[c for c in id_cols if c in df.columns])
        return df


class EnsembleLogisticRegressorModel(SupervisedModel):
    """
    A wrapper class to train and validate ensemble xgboost models.
    """

    def __init__(
        self,
        data: EnsembleDataloader,
        params: LogisticRegressionHyperparamConfig,
        cv: CrossValidationConfig | None,
        tracker: Tracker,
    ) -> None:
        super().__init__(data, params, cv, tracker)
        self.models: dict[int, LogisticRegression] = {}
        self.scores: dict[int, tuple[float, float]] = {}

    def fit(self, test_season: int, start_season: int = 2003) -> None:
        """
        Fit the model on data provided by the given dataloader.

        The model is using data up to but not including the specified season.

        Args:
            test_season (int): Season for which the model should be validated.
            start_season (int): Start season for which the model should be trained on.
        """
        self.data.setup()
        for idx, (valid_season, (X_train, y_train), (X_valid, y_valid)) in enumerate(
            self.data.train_data(test_season, start_season)
        ):
            X_train = self._drop_and_sort_features(X_train)
            X_valid = self._drop_and_sort_features(X_valid)

            if not idx:
                self._fit_scaler(X_train)
            X_train = self._transform(X_train)
            X_valid = self._transform(X_valid)

            self.models[valid_season] = LogisticRegression(**self.params.as_params())

            y_binary = (y_train.values >= 0.5).astype(int)
            self.models[valid_season].fit(X_train, y_binary)

            train_preds = np.clip(self.models[valid_season].predict_proba(X_train)[:, 1], 0, 1)
            valid_preds = np.clip(self.models[valid_season].predict_proba(X_valid)[:, 1], 0, 1)
            self.scores[valid_season] = (
                brier_score(y_train.values, train_preds),
                brier_score(y_valid.values, valid_preds),
            )
            self.tracker.log({Metrics.TRAIN_ENSEMBLE_BRIER: self.scores[valid_season][0]}, step=valid_season)
            self.tracker.log({Metrics.VALID_ENSEMBLE_BRIER: self.scores[valid_season][1]}, step=valid_season)
        self.tracker.log({Metrics.TRAIN_BRIER: np.mean([scores[0] for scores in self.scores.values()])})
        self.tracker.log({Metrics.VALID_BRIER: np.mean([scores[1] for scores in self.scores.values()])})

    def validate(self):
        """Validation is done during fit for ensemble models."""
        pass

    def predict(self, matchups: pd.DataFrame) -> Iterable[float]:
        """
        Make predictions for the given matchups.

        Args:
            matchups (pd.DataFrame): Matchups to make predictions for.
        Note:
            Fit must be called first.
        """
        assert self.models, "Call fit() before predict()"
        X = self.data.test_data(matchups)
        X = self._drop_and_sort_features(X)
        X = self._transform(X)

        preds = [np.clip(model.predict_proba(X)[:, 1], 0, 1) for model in self.models.values()]
        preds = np.mean(preds, axis=0)
        return pd.Series(preds, index=X.index)

    @staticmethod
    def _drop_and_sort_features(df: pd.DataFrame) -> pd.DataFrame:
        id_cols = [Columns.SEASON, Columns.T1_TEAM_ID, Columns.T2_TEAM_ID]
        df = df.drop(columns=[c for c in id_cols if c in df.columns])
        return df
