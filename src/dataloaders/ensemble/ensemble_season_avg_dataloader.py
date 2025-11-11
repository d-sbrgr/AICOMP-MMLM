from collections.abc import Generator

import pandas as pd

from src.utils.constants import Columns

from ..base_dataloader import EnsembleDataloader
from ..base_season_average import BaseSeasonAverage
from ..feature_selection import FeatureSelection


class EnsembleSeasonAverageDataLoader(FeatureSelection, BaseSeasonAverage, EnsembleDataloader):
    """
    Loads and prepares season-averaged features and metadata for XGBoost and random forest modeling.

    This class computes per-team season averages, merges additional features (ELO, streaks, seeds, quality),
    and attaches the last ELO for each team in the regular season. It provides train and test data
    for model fitting and evaluation.

    Attributes:
        _data (pd.DataFrame): Final tournament data with all merged features.
        _df_season_stats_t1 (pd.DataFrame): Season averages and last ELO for T1 perspective.
        _df_season_stats_t2 (pd.DataFrame): Season averages and last ELO for T2 perspective.
        _df_seeds_t1 (pd.DataFrame): Seed info for T1 teams.
        _df_seeds_t2 (pd.DataFrame): Seed info for T2 teams.
        _df_quality_t1 (pd.DataFrame): Quality info for T1 teams.
        _df_quality_t2 (pd.DataFrame): Quality info for T2 teams.
        _features (list[str]): List of feature columns used for modeling.
        _prediction_season (int): The season for which predictions are made.
        _start_season (int): The first season for which the model is trained on.
    """

    def __init__(self, num_features: int):
        EnsembleDataloader.__init__(self)
        FeatureSelection.__init__(self, num_features)
        BaseSeasonAverage.__init__(self)
        self._data: pd.DataFrame | None = None
        self._df_season_stats_t1: pd.DataFrame | None = None
        self._df_season_stats_t2: pd.DataFrame | None = None
        self._df_seeds_t1: pd.DataFrame | None = None
        self._df_seeds_t2: pd.DataFrame | None = None
        self._df_quality_t1: pd.DataFrame | None = None
        self._df_quality_t2: pd.DataFrame | None = None
        self._prediction_season: int | None = None
        self._start_season: int | None = None

    def setup(self) -> None:
        """
        Load, merge, and prepare all regular season and tournament data, compute season averages,
        attach last ELO, seeds, and quality features for each team, and store the final tournament data.
        """
        super().setup()

    def train_data(
        self, prediction_season: int, start_season: int
    ) -> Generator[tuple[int, tuple[pd.DataFrame, pd.Series], ...], None, None]:
        """
        Generator yielding training and validation data for season cross-validation.

        One generator yield per target season lower than prediction_season and greater than or equal to start_season.

        Args:
            prediction_season (int): The season to predict (excluded from training).
            start_season (int): The first season contained in the training data.

        Yields:
            tuple[int, tuple[pd.DataFrame, pd.Series], ...]: Target season, training data and validation data.
        """
        if prediction_season <= start_season:
            raise ValueError("prediction_season must be greater than minimum_season")
        if not 2003 <= prediction_season <= 2025:
            raise ValueError("prediction_season must be between 2003 and 2025")
        if not 2003 <= start_season <= 2024:
            raise ValueError("minimum_season must be between 2003 and 2024")
        self._prediction_season = prediction_season
        self._start_season = start_season
        for season in range(start_season, prediction_season):
            if self._data[self._data[Columns.SEASON] == season].empty:
                print(f"Warning: No data for season {season}, skipping fold.")
                continue
            train_mask = (
                (self._data[Columns.SEASON] < self._prediction_season)
                & (self._data[Columns.SEASON] >= self._start_season)
                & (self._data[Columns.SEASON] != season)
            )
            valid_mask = self._data[Columns.SEASON] == season
            yield (
                season,
                (
                    self._data.loc[train_mask, self._features],
                    self._data.loc[train_mask, Columns.TARGET].squeeze(),
                ),
                (
                    self._data.loc[valid_mask, self._features],
                    self._data.loc[valid_mask, Columns.TARGET].squeeze(),
                ),
            )

    def test_data(self, df_matchups: pd.DataFrame) -> pd.DataFrame:
        """
        Extend a DataFrame of matchups with all engineered features for prediction.

        Args:
            df_matchups (pd.DataFrame): DataFrame of matchups to extend.

        Returns:
            pd.DataFrame: DataFrame with all features for prediction.
        """
        if not self._features:
            raise RuntimeError("Must call train_data first")
        df = self._extend_matchups_with_features(df_matchups)
        return df[self._features]
