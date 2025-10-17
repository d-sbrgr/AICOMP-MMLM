import pandas as pd

from ..datasets.datasets import (
    detailed_regular_season_results,
    detailed_tourney_results,
    overall_elo_delta,
    regular_season_streaks,
    seeds,
    team_quality,
)
from ..utils import Columns
from .base_dataloader import BaseDataloader


class SeasonAverageDataLoader(BaseDataloader):
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
    """

    def __init__(self, features: list[str]):
        self._data: pd.DataFrame | None = None
        self._df_season_stats_t1: pd.DataFrame | None = None
        self._df_season_stats_t2: pd.DataFrame | None = None
        self._df_seeds_t1: pd.DataFrame | None = None
        self._df_seeds_t2: pd.DataFrame | None = None
        self._df_quality_t1: pd.DataFrame | None = None
        self._df_quality_t2: pd.DataFrame | None = None
        self._features: list[str] = features
        self._prediction_season: int | None = None

    @property
    def features(self) -> list[str]:
        return self._features

    @features.setter
    def features(self, features: list[str]):
        self._features = features

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
            on=["Season", "DayNum", "T1_TeamID", "T2_TeamID"],
            how="left",
        )
        df_regular = pd.merge(
            df_regular,
            self._prepare_additional_features(regular_season_streaks()),
            on=["Season", "DayNum", "T1_TeamID", "T2_TeamID"],
            how="left",
        )

        # Compute season averages for each team
        boxcols = list(
            set(df_regular.columns).difference(
                {"Season", "DayNum", "T1_TeamID", "T2_TeamID", "NumOT", "MenWomen", "Target"}
            )
        )
        df_season_stats = df_regular.groupby(["Season", "T1_TeamID"])[boxcols].agg("mean").reset_index()

        # Stack T1 and T2 ELOs for each team/game
        df_elo = df_regular[["Season", "DayNum", "T1_TeamID", "T1_Elo"]].rename(
            columns={"T1_TeamID": "TeamID", "T1_Elo": "LastElo"}
        )
        df_elo = df_elo.sort_values(["Season", "TeamID", "DayNum"]).groupby(["Season", "TeamID"]).tail(1)

        # Merge last ELO into season stats
        df_season_stats = pd.merge(
            df_season_stats, df_elo, left_on=["Season", "T1_TeamID"], right_on=["Season", "TeamID"], how="left"
        )
        df_season_stats = df_season_stats.drop(columns=["TeamID"])
        df_season_stats = df_season_stats.rename(columns={"LastElo": "T1_LastElo"})

        # Prepare team stats for merging (T1 and T2 perspectives)
        self._df_season_stats_T1 = df_season_stats.copy()
        self._df_season_stats_T1.columns = [
            "T1_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(self._df_season_stats_T1.columns)
        ]
        self._df_season_stats_T1 = self._df_season_stats_T1.rename(
            {"T1_avg_Season": "Season", "T1_avg_TeamID": "T1_TeamID", "T1_avg_LastElo": "T1_Elo"}, axis=1
        )

        self._df_season_stats_T2 = df_season_stats.copy()
        self._df_season_stats_T2.columns = [
            "T2_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(self._df_season_stats_T2.columns)
        ]
        self._df_season_stats_T2 = self._df_season_stats_T2.rename(
            {"T2_avg_Season": "Season", "T2_avg_TeamID": "T2_TeamID", "T2_avg_LastElo": "T2_Elo"}, axis=1
        )

        # Prepare and merge seed data
        df_seeds = seeds()
        self._df_seeds_T1 = df_seeds[["Season", "TeamID", "Seed"]].copy()
        self._df_seeds_T2 = df_seeds[["Season", "TeamID", "Seed"]].copy()
        self._df_seeds_T1.columns = ["Season", "T1_TeamID", "T1_seed"]
        self._df_seeds_T2.columns = ["Season", "T2_TeamID", "T2_seed"]

        # Select relevant columns and add seed difference
        df_tourney = self._prepare_detailed_results(detailed_tourney_results())
        df_tourney = df_tourney[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "Target", "MenWomen"]]
        df_tourney = pd.merge(df_tourney, self._df_seeds_T1, on=["Season", "T1_TeamID"], how="left")
        df_tourney = pd.merge(df_tourney, self._df_seeds_T2, on=["Season", "T2_TeamID"], how="left")
        df_tourney["SeedDiff"] = df_tourney["T2_seed"] - df_tourney["T1_seed"]

        # Merge with team stats
        df_tourney = pd.merge(df_tourney, self._df_season_stats_T1, on=["Season", "T1_TeamID"], how="left")
        df_tourney = pd.merge(df_tourney, self._df_season_stats_T2, on=["Season", "T2_TeamID"], how="left")

        # Prepare and merge team quality data
        df_quality = team_quality()
        self._df_quality_t1 = df_quality.copy()
        self._df_quality_t1.columns = [
            x.replace("Quality", "T1_Quality").replace("TeamID", "T1_TeamID") for x in list(self._df_quality_t1.columns)
        ]
        self._df_quality_t2 = df_quality.copy()
        self._df_quality_t2.columns = [
            x.replace("Quality", "T2_Quality").replace("TeamID", "T2_TeamID") for x in list(self._df_quality_t2.columns)
        ]
        df_tourney = pd.merge(df_tourney, self._df_quality_t1, on=["Season", "T1_TeamID"], how="left")
        df_tourney = pd.merge(df_tourney, self._df_quality_t2, on=["Season", "T2_TeamID"], how="left")
        df_tourney["QualityDiff"] = df_tourney["T2_Quality"] - df_tourney["T1_Quality"]

        # Store final tournament data
        self._data = df_tourney

    def train_data(self, prediction_season: int, featues: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
        """
        Return training features and targets for all seasons before the prediction season.

        Args:
            prediction_season (int): The season to predict (excluded from training).
            featues (list[str] | None): List of feature columns to use. If None, uses all available features.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Training features and target values.
        """
        if not 2003 <= prediction_season <= 2025:
            raise ValueError("prediction_season must be between 2003 and 2025")
        self._prediction_season = prediction_season
        return (
            self._data.loc[self._data[Columns.SEASON] < prediction_season, self._features],
            self._data.loc[self._data[Columns.SEASON] < prediction_season, "Target"].squeeze(),
        )

    def valid_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Return validation features and targets for the prediction season.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Validation features and target values.
        """
        if not self._features:
            raise RuntimeError("Must call train_data first")
        return (
            self._data.loc[self._data[Columns.SEASON] == self._prediction_season, self._features],
            self._data.loc[self._data[Columns.SEASON] == self._prediction_season, "Target"].squeeze(),
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
        df["Season"] = df["ID"].apply(lambda t: int(t.split("_")[0]))
        df.columns = [x.replace("Lower", "T1_Team").replace("Higher", "T2_Team") for x in list(df.columns)]
        df["MenWomen"] = df["T1_TeamID"].apply(lambda t: 0 if str(t)[0] == "1" else 1)
        df = pd.merge(df, self._df_season_stats_T1, on=["Season", "T1_TeamID"], how="left")
        df = pd.merge(df, self._df_season_stats_T2, on=["Season", "T2_TeamID"], how="left")
        df = pd.merge(df, self._df_seeds_T1, on=["Season", "T1_TeamID"], how="left")
        df = pd.merge(df, self._df_seeds_T2, on=["Season", "T2_TeamID"], how="left")
        df = pd.merge(df, self._df_quality_t1, on=["Season", "T1_TeamID"], how="left")
        df = pd.merge(df, self._df_quality_t2, on=["Season", "T2_TeamID"], how="left")
        df["T1_seed"] = df["T1_seed"].fillna(32)
        df["T2_seed"] = df["T2_seed"].fillna(32)
        df["PointDiff"] = df["T1_avg_Score"] - df["T2_avg_Score"]
        df["SeedDiff"] = df["T2_seed"] - df["T1_seed"]
        df["QualityDiff"] = df["T2_Quality"] - df["T1_Quality"]
        return df

    def _prepare_detailed_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare detailed results by normalizing stats, swapping team perspectives,
        and computing point differentials and targets.
        """
        df = df[list(set(df.columns).difference({"WLoc"}))]

        adjot = (40 + 5 * df["NumOT"]) / 40
        adjcols = list(set(df.columns).difference({"DayNum", "LTeamID", "NumOT", "Season", "WTeamID"}))
        for col in adjcols:
            df[col] = df[col] / adjot

        dfswap = df.copy()
        df.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in list(df.columns)]
        dfswap.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in list(dfswap.columns)]
        output = pd.concat([df, dfswap]).reset_index(drop=True)
        output["PointDiff"] = output["T1_Score"] - output["T2_Score"]
        output["Target"] = (output["PointDiff"] > 0) * 1
        output["MenWomen"] = (output["T1_TeamID"].apply(lambda t: str(t).startswith("1"))) * 1  # 0: women, 1: men
        return output

    def _prepare_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and swap additional features (ELO, streaks, quality) for merging into main data.
        Computes feature differences for each matchup.
        """
        columns = set(df.columns)
        columns &= {
            "Season",
            "DayNum",
            "LTeamID",
            "WTeamID",
            "WElo",
            "LElo",
            "WEloDelta",
            "LEloDelta",
            "WStreak",
            "LStreak",
        }
        df = df[list(columns)]

        dfswap = df.copy()
        df.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in list(df.columns)]
        dfswap.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in list(dfswap.columns)]
        output = pd.concat([df, dfswap]).reset_index(drop=True)
        if any("Elo" in col for col in output.columns):
            output["EloDiff"] = output["T1_Elo"] - output["T2_Elo"]
            output["EloDeltaDiff"] = output["T1_EloDelta"] - output["T2_EloDelta"]
        if any("Streak" in col for col in output.columns):
            output["StreakDiff"] = output["T1_Streak"] - output["T2_Streak"]
        if any("Quality" in col for col in output.columns):
            output["QualityDiff"] = output["T1_Quality"] - output["T2_Quality"]
        return output
