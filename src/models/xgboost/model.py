from collections.abc import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb

from ...dataloaders.base_dataloader import BaseDataloader
from ...evaluation import brier_score
from ...experiments import Tracker
from ..model import Model
from .config import CrossValidationConfig, XGBHyperparamConfig
from .cv_runner import CVRunner


class XGBRegressorModel(Model):
    """
    A wrapper class to train and validate xgboost models.
    """

    def __init__(
        self, data: BaseDataloader, params: XGBHyperparamConfig, cv: CrossValidationConfig, tracker: Tracker
    ) -> None:
        super().__init__()
        self.data = data
        self.params = params
        self.cv_cfg = cv
        self.tracker = tracker
        self.model: xgb.Booster | None = None

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

        cv_runner = CVRunner(self.cv_cfg)
        splits = cv_runner.split(y)
        out_of_frame = np.zeros(len(y))

        for fold, (tr_idx, va_idx) in enumerate(splits):
            dtrain = xgb.DMatrix(X.iloc[tr_idx], label=y.values[tr_idx])
            dvalid = xgb.DMatrix(X.iloc[va_idx], label=y.values[va_idx])
            evallist = [(dtrain, "train"), (dvalid, "valid")]
            booster = xgb.train(
                self.params.as_params(),
                dtrain,
                num_boost_round=self.params.num_rounds,
                evals=evallist,
                verbose_eval=False,
            )
            preds = booster.predict(dvalid)
            out_of_frame[va_idx] = preds
            self.tracker.log({"fold": fold, "train_brier_cv_fold": brier_score(y.values[va_idx], preds)})
        self.tracker.log({"train_brier_cv_full": brier_score(y.values, out_of_frame)})

        dtrain = xgb.DMatrix(X, label=y.values)
        self.model = xgb.train(
            self.params.as_params(), dtrain, num_boost_round=self.params.num_rounds, verbose_eval=False
        )
        preds = self.model.predict(dtrain)
        self.tracker.log({"train_brier": brier_score(y.values, preds)})

    def validate(self):
        """
        Validate the model for the season provided in `fit()`.

        Note:
            Fit must be called beforehand
        """
        assert self.model is not None, "Call fit() before predict()"
        X, y = self.data.valid_data()
        X = self._drop_and_sort_features(X)
        dvalid = xgb.DMatrix(X)
        preds = self.model.predict(dvalid)
        self.tracker.log({"valid_brier": brier_score(y.values, preds)})

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
        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)
        return pd.Series(preds, index=X.index)

    def _drop_and_sort_features(self, df: pd.DataFrame) -> pd.DataFrame:
        id_cols = ["Season", "T1_TeamID", "T2_TeamID"]
        df = df.drop(columns=[c for c in id_cols if c in df.columns])
        return df
