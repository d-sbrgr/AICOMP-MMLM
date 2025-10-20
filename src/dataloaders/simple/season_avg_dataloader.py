import pandas as pd

from src.datasets.datasets import (
    detailed_regular_season_results,
    detailed_tourney_results,
    overall_elo_delta,
    regular_season_streaks,
    seeds,
    team_quality,
)
from src.utils.constants import Columns

from .feature_selection_dataloader import FeatureSelectionDataLoader


class SeasonAverageDataLoader(FeatureSelectionDataLoader):
    """
    Loads and prepares season-averaged features and metadata for XGBoost and random forest modeling.

    This class computes per-team season averages, merges additional features (ELO, streaks, seeds, quality),
    and attaches the last ELO for each team in the regular season. It provides train, validation, and test data
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
        super().__init__(num_features)
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
        # Prepare regular season data with detailed results and additional features
        df_regular = self._prepare_detailed_results(detailed_regular_season_results())
        df_regular = pd.merge(
            df_regular,
            self._prepare_additional_features(overall_elo_delta()),
            on=[Columns.SEASON, Columns.DAY_NUM, Columns.T1_TEAM_ID, Columns.T2_TEAM_ID],
            how="left",
        )
        df_regular = pd.merge(
            df_regular,
            self._prepare_additional_features(regular_season_streaks()),
            on=[Columns.SEASON, Columns.DAY_NUM, Columns.T1_TEAM_ID, Columns.T2_TEAM_ID],
            how="left",
        )

        # Compute season averages for each team
        boxcols = list(
            set(df_regular.columns).difference(
                {
                    Columns.SEASON,
                    Columns.DAY_NUM,
                    Columns.T1_TEAM_ID,
                    Columns.T2_TEAM_ID,
                    Columns.NUM_OT,
                    Columns.MEN_WOMEN,
                    Columns.TARGET,
                }
            )
        )
        df_season_stats = df_regular.groupby([Columns.SEASON, Columns.T1_TEAM_ID])[boxcols].agg("mean").reset_index()

        # Stack T1 and T2 ELOs for each team/game
        df_elo = df_regular[[Columns.SEASON, Columns.DAY_NUM, Columns.T1_TEAM_ID, Columns.T1_ELO]].rename(
            columns={Columns.T1_TEAM_ID: Columns.TEAM_ID, Columns.T1_ELO: Columns.LAST_ELO}
        )
        df_elo = (
            df_elo.sort_values([Columns.SEASON, Columns.TEAM_ID, Columns.DAY_NUM])
            .groupby([Columns.SEASON, Columns.TEAM_ID])
            .tail(1)
        )

        # Merge last ELO into season stats
        df_season_stats = pd.merge(
            df_season_stats,
            df_elo,
            left_on=[Columns.SEASON, Columns.T1_TEAM_ID],
            right_on=[Columns.SEASON, Columns.TEAM_ID],
            how="left",
        )
        df_season_stats = df_season_stats.drop(columns=[Columns.TEAM_ID])
        df_season_stats = df_season_stats.rename(columns={Columns.LAST_ELO: "T1_LastElo"})

        # Prepare team stats for merging (T1 and T2 perspectives)
        self._df_season_stats_T1 = df_season_stats.copy()
        self._df_season_stats_T1.columns = [
            "T1_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(self._df_season_stats_T1.columns)
        ]
        self._df_season_stats_T1 = self._df_season_stats_T1.rename(
            {"T1_avg_Season": Columns.SEASON, "T1_avg_TeamID": Columns.T1_TEAM_ID, "T1_avg_LastElo": Columns.T1_ELO},
            axis=1,
        )

        self._df_season_stats_T2 = df_season_stats.copy()
        self._df_season_stats_T2.columns = [
            "T2_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(self._df_season_stats_T2.columns)
        ]
        self._df_season_stats_T2 = self._df_season_stats_T2.rename(
            {"T2_avg_Season": Columns.SEASON, "T2_avg_TeamID": Columns.T2_TEAM_ID, "T2_avg_LastElo": Columns.T2_ELO},
            axis=1,
        )

        # Prepare and merge seed data
        df_seeds = seeds()
        self._df_seeds_T1 = df_seeds[[Columns.SEASON, Columns.TEAM_ID, Columns.SEED]].copy()
        self._df_seeds_T2 = df_seeds[[Columns.SEASON, Columns.TEAM_ID, Columns.SEED]].copy()
        self._df_seeds_T1.columns = [Columns.SEASON, Columns.T1_TEAM_ID, Columns.T1_SEED]
        self._df_seeds_T2.columns = [Columns.SEASON, Columns.T2_TEAM_ID, Columns.T2_SEED]

        # Select relevant columns and add seed difference
        df_tourney = self._prepare_detailed_results(detailed_tourney_results())
        df_tourney = df_tourney[
            [
                Columns.SEASON,
                Columns.T1_TEAM_ID,
                Columns.T2_TEAM_ID,
                Columns.POINT_DIFF,
                Columns.TARGET,
                Columns.MEN_WOMEN,
            ]
        ]
        df_tourney = pd.merge(df_tourney, self._df_seeds_T1, on=[Columns.SEASON, Columns.T1_TEAM_ID], how="left")
        df_tourney = pd.merge(df_tourney, self._df_seeds_T2, on=[Columns.SEASON, Columns.T2_TEAM_ID], how="left")
        df_tourney[Columns.SEED_DIFF] = df_tourney[Columns.T2_SEED] - df_tourney[Columns.T1_SEED]

        # Merge with team stats
        df_tourney = pd.merge(df_tourney, self._df_season_stats_T1, on=[Columns.SEASON, Columns.T1_TEAM_ID], how="left")
        df_tourney = pd.merge(df_tourney, self._df_season_stats_T2, on=[Columns.SEASON, Columns.T2_TEAM_ID], how="left")

        # Prepare and merge team quality data
        df_quality = team_quality()
        self._df_quality_t1 = df_quality.copy()
        self._df_quality_t1.columns = [
            x.replace(Columns.QUALITY, Columns.T1_QUALITY).replace(Columns.TEAM_ID, Columns.T1_TEAM_ID)
            for x in list(self._df_quality_t1.columns)
        ]
        self._df_quality_t2 = df_quality.copy()
        self._df_quality_t2.columns = [
            x.replace(Columns.QUALITY, Columns.T2_QUALITY).replace(Columns.TEAM_ID, Columns.T2_TEAM_ID)
            for x in list(self._df_quality_t2.columns)
        ]
        df_tourney = pd.merge(df_tourney, self._df_quality_t1, on=[Columns.SEASON, Columns.T1_TEAM_ID], how="left")
        df_tourney = pd.merge(df_tourney, self._df_quality_t2, on=[Columns.SEASON, Columns.T2_TEAM_ID], how="left")
        df_tourney[Columns.QUALITY_DIFF] = df_tourney[Columns.T2_QUALITY] - df_tourney[Columns.T1_QUALITY]

        # Store final tournament data
        self._data = df_tourney

    def train_data(self, prediction_season: int, start_season: int) -> tuple[pd.DataFrame, pd.Series]:
        """
        Return training features and targets for all seasons before the prediction season.

        Args:
            prediction_season (int): The season to predict (excluded from training).
            start_season (int): The first season contained in the training data.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Training features and target values.
        """
        if prediction_season <= start_season:
            raise ValueError("prediction_season must be greater than minimum_season")
        if not 2003 <= prediction_season <= 2025:
            raise ValueError("prediction_season must be between 2003 and 2025")
        if not 2003 <= start_season <= 2024:
            raise ValueError("minimum_season must be between 2003 and 2024")
        self._prediction_season = prediction_season
        self._start_season = start_season
        mask = (self._data[Columns.SEASON] < self._prediction_season) & (
            self._data[Columns.SEASON] >= self._start_season
        )
        return (
            self._data.loc[mask, self._features],
            self._data.loc[mask, Columns.TARGET].squeeze(),
        )

    def valid_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Return validation features and targets for the prediction season.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Validation features and target values.
        """
        if not self._features:
            raise RuntimeError("Must call train_data first")
        mask = (self._data[Columns.SEASON] == self._prediction_season) & (
            self._data[Columns.SEASON] >= self._start_season
        )
        return (
            self._data.loc[mask, self._features],
            self._data.loc[mask, Columns.TARGET].squeeze(),
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

    def _extend_matchups_with_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all engineered features to a matchups DataFrame for prediction.
        Includes team stats, seeds, quality, and derived features.
        """
        df[Columns.SEASON] = df[Columns.ID].apply(lambda t: int(t.split("_")[0]))
        df.columns = [x.replace("Lower", "T1_Team").replace("Higher", "T2_Team") for x in list(df.columns)]
        df[Columns.MEN_WOMEN] = df[Columns.T1_TEAM_ID].apply(lambda t: 0 if str(t)[0] == "1" else 1)
        df = pd.merge(df, self._df_season_stats_T1, on=[Columns.SEASON, Columns.T1_TEAM_ID], how="left")
        df = pd.merge(df, self._df_season_stats_T2, on=[Columns.SEASON, Columns.T2_TEAM_ID], how="left")
        df = pd.merge(df, self._df_seeds_T1, on=[Columns.SEASON, Columns.T1_TEAM_ID], how="left")
        df = pd.merge(df, self._df_seeds_T2, on=[Columns.SEASON, Columns.T2_TEAM_ID], how="left")
        df = pd.merge(df, self._df_quality_t1, on=[Columns.SEASON, Columns.T1_TEAM_ID], how="left")
        df = pd.merge(df, self._df_quality_t2, on=[Columns.SEASON, Columns.T2_TEAM_ID], how="left")
        df[Columns.T1_SEED] = df[Columns.T1_SEED].fillna(32)
        df[Columns.T2_SEED] = df[Columns.T2_SEED].fillna(32)
        df[Columns.POINT_DIFF] = df["T1_avg_Score"] - df["T2_avg_Score"]
        df[Columns.SEED_DIFF] = df[Columns.T2_SEED] - df[Columns.T1_SEED]
        df[Columns.QUALITY_DIFF] = df[Columns.T2_QUALITY] - df[Columns.T1_QUALITY]
        return df

    def _prepare_detailed_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare detailed results by normalizing stats, swapping team perspectives,
        and computing point differentials and targets.
        """
        df = df[list(set(df.columns).difference({Columns.WLOC}))]

        adjot = (40 + 5 * df[Columns.NUM_OT]) / 40
        adjcols = list(
            set(df.columns).difference(
                {Columns.DAY_NUM, Columns.LTEAM_ID, Columns.NUM_OT, Columns.SEASON, Columns.WTEAM_ID}
            )
        )
        for col in adjcols:
            df[col] = df[col] / adjot

        dfswap = df.copy()
        df.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in list(df.columns)]
        dfswap.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in list(dfswap.columns)]
        output = pd.concat([df, dfswap]).reset_index(drop=True)
        output[Columns.POINT_DIFF] = output["T1_Score"] - output["T2_Score"]
        output[Columns.TARGET] = (output[Columns.POINT_DIFF] > 0) * 1
        output[Columns.MEN_WOMEN] = (
            output[Columns.T1_TEAM_ID].apply(lambda t: str(t).startswith("1"))
        ) * 1  # 0: women, 1: men
        return output

    def _prepare_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and swap additional features (ELO, streaks, quality) for merging into main data.
        Computes feature differences for each matchup.
        """
        columns = set(df.columns)
        columns &= {
            Columns.SEASON,
            Columns.DAY_NUM,
            Columns.LTEAM_ID,
            Columns.WTEAM_ID,
            Columns.WELO,
            Columns.LELO,
            Columns.WELO_DELTA,
            Columns.LELO_DELTA,
            "WStreak",
            "LStreak",
        }
        df = df[list(columns)]

        dfswap = df.copy()
        df.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in list(df.columns)]
        dfswap.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in list(dfswap.columns)]
        output = pd.concat([df, dfswap]).reset_index(drop=True)
        if any(Columns.ELO in col for col in output.columns):
            output[Columns.ELO_DIFF] = output[Columns.T1_ELO] - output[Columns.T2_ELO]
            output[Columns.ELO_DELTA_DIFF] = output["T1_EloDelta"] - output["T2_EloDelta"]
        if any(Columns.STREAK in col for col in output.columns):
            output[Columns.STREAK_DIFF] = output[Columns.T1_STREAK] - output[Columns.T2_STREAK]
        if any(Columns.QUALITY in col for col in output.columns):
            output[Columns.QUALITY_DIFF] = output[Columns.T1_QUALITY] - output[Columns.T2_QUALITY]
        return output
