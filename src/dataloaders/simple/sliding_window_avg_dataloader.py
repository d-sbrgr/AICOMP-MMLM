import pandas as pd
from sklearn.model_selection import train_test_split

from src.datasets.datasets import detailed_regular_season_results, overall_elo_delta, sliding_window_average
from src.features.quality import fast_compute_quality
from src.utils.constants import Columns

from ..base_dataloader import BaseDataloader
from ..common import generate_perspectives, prepare_detailed_results, prepare_engineered_features
from ..feature_selection import FeatureSelection

MAX_REGULAR_SEASON_DAY = 132


class SlidingWindowAvgDataLoader(FeatureSelection, BaseDataloader):
    """
    Loads and prepares weighted sliding-window averaged features for modeling.

    This dataloader creates matchup records from ALL games (regular season + tournament),
    computes weighted aggregations of team stats in a sliding window of the last season-long amount of games,
    and generates a dataset where each game creates two records (Team A vs Team B, and Team B vs Team A).
    The data is then shuffled and split into train/validation/test sets.

    Key differences from SeasonAverageDataLoader:
    - Uses ALL games (regular + tournament) as training data, not just tournament games
    - Applies time-based discounting to weigh recent games higher
    - Creates bidirectional matchup records
    - Random train/val split instead of temporal split
    """

    def __init__(
        self,
        num_features: int,
        train_split: float = 0.75,
        valid_split: float = 0.25,
        random_seed: int = 42,
    ):
        """
        Initialize the SlidingWindowAvgDataLoader.

        Args:
            num_features (int): Number of top features to select for modeling.
            train_split (float): Fraction of data for training (default: 0.75).
            valid_split (float): Fraction of data for validation (default: 0.25).
                                Remaining data becomes test set.
            random_seed (int): Random seed for reproducible splits (default: 42).
        """
        FeatureSelection.__init__(self, num_features)
        BaseDataloader.__init__(self)
        self._data: pd.DataFrame | None = None
        self._train_data: pd.DataFrame | None = None
        self._valid_data: pd.DataFrame | None = None

        # Split parameters
        self._train_split = train_split
        self._valid_split = valid_split
        self._random_seed = random_seed

        # Validate split parameters
        assert 0 < train_split < 1, "train_split must be between 0 and 1"
        assert 0 < valid_split < 1, "valid_split must be between 0 and 1"
        assert train_split + valid_split == 1, "train_split + valid_split must be equal to 1"

    def setup(self) -> None:
        """
        Load data from prebuilt sliding window average dataset
        """
        self._data = sliding_window_average()
        self._split_data()

    def train_data(self, prediction_season: int, start_season: int) -> tuple[pd.DataFrame, pd.Series]:
        """
        Return training features and targets.

        Note: prediction_season and start_season are kept for API compatibility
        but are not used since this dataloader uses random splits, not temporal splits.

        Args:
            prediction_season (int): Ignored. Kept for compatibility.
            start_season (int): Ignored. Kept for compatibility.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Training features and target values.
        """
        if self._train_data is None:
            raise RuntimeError("Must call setup() before accessing train_data")

        return (
            self._train_data[self._features],
            self._train_data[Columns.TARGET].squeeze(),
        )

    def valid_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Return validation features and targets.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Validation features and target values.
        """
        if self._valid_data is None:
            raise RuntimeError("Must call setup() before accessing valid_data")

        return (
            self._valid_data[self._features],
            self._valid_data[Columns.TARGET].squeeze(),
        )

    def test_data(self, df_matchups: pd.DataFrame = None) -> pd.DataFrame:
        """
        Return test features for evaluation or extend matchups with features for prediction.

        Args:
            df_matchups (pd.DataFrame, optional): If provided, extends these matchups with features.
                                                 If None, returns the test split features.

        Returns:
            pd.DataFrame: Test features or extended matchup features.
        """
        if not self._features:
            raise RuntimeError("Must call setup first to set features")
        df = self._extend_matchups_with_features(df_matchups)
        return df[self._features]

    def _extend_matchups_with_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all engineered features to a matchups DataFrame for prediction.

        Args:
            df (pd.DataFrame): Matchups DataFrame with columns like ID, LowerID, HigherID.

        Returns:
            pd.DataFrame: DataFrame with all features added.
        """
        df[Columns.SEASON] = df[Columns.ID].apply(lambda t: int(t.split("_")[0]))
        df.columns = [x.replace("Lower", "T1_Team").replace("Higher", "T2_Team") for x in list(df.columns)]
        df[Columns.MEN_WOMEN] = df[Columns.T1_TEAM_ID].apply(lambda t: 0 if str(t)[0] == "1" else 1)

        _df_season_stats_T1, _df_season_stats_T2 = self._get_stats_for_season(df[Columns.SEASON].unique()[0])

        return self._merge_stats_with_matchups(df, _df_season_stats_T1, _df_season_stats_T2)

    def _get_stats_for_season(self, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_regular = prepare_detailed_results(detailed_regular_season_results())
        elo_delta = prepare_engineered_features(overall_elo_delta())
        df = pd.merge(
            df_regular,
            elo_delta,
            on=[Columns.SEASON, Columns.DAY_NUM, Columns.T1_TEAM_ID, Columns.T2_TEAM_ID],
            how="left",
        )

        data = df[df[Columns.SEASON] == season]
        stats = self._compute_weighted_season_stats(self._apply_time_discount(data, 0.98))
        team_qualities = fast_compute_quality(data)
        stats = pd.merge(stats, team_qualities, how="left", on=Columns.TEAM_ID)
        _df_season_stats_T1, _df_season_stats_T2 = generate_perspectives(stats)

        df_elo: pd.DataFrame = elo_delta[
            [Columns.SEASON, Columns.DAY_NUM, Columns.T1_TEAM_ID, Columns.T1_ELO, Columns.T1_ELO_DELTA]
        ].rename(
            columns={
                Columns.T1_TEAM_ID: Columns.TEAM_ID,
                Columns.T1_ELO: Columns.ELO,
                Columns.T1_ELO_DELTA: Columns.ELO_DELTA,
            }
        )
        df_elo = df_elo[df_elo[Columns.SEASON] == season].copy()
        df_elo = df_elo.sort_values([Columns.TEAM_ID, Columns.DAY_NUM]).groupby([Columns.TEAM_ID]).last().reset_index()

        _df_season_stats_T1 = pd.merge(
            _df_season_stats_T1,
            df_elo,
            left_on=[Columns.T1_TEAM_ID],
            right_on=[Columns.TEAM_ID],
            how="left",
        )
        _df_season_stats_T2 = pd.merge(
            _df_season_stats_T2,
            df_elo,
            left_on=[Columns.T2_TEAM_ID],
            right_on=[Columns.TEAM_ID],
            how="left",
        )
        return (
            _df_season_stats_T1.rename(columns={Columns.ELO: Columns.T1_ELO, Columns.ELO_DELTA: Columns.T1_ELO_DELTA}),
            _df_season_stats_T2.rename(columns={Columns.ELO: Columns.T2_ELO, Columns.ELO_DELTA: Columns.T2_ELO_DELTA}),
        )

    @staticmethod
    def _merge_stats_with_matchups(
        df_matchups: pd.DataFrame, df_T1_stats: pd.DataFrame, df_T2_stats: pd.DataFrame
    ) -> pd.DataFrame:
        df_matchups = pd.merge(df_matchups, df_T1_stats, on=[Columns.SEASON, Columns.T1_TEAM_ID], how="left")
        df_matchups = pd.merge(df_matchups, df_T2_stats, on=[Columns.SEASON, Columns.T2_TEAM_ID], how="left")

        df_matchups[Columns.POINT_DIFF] = df_matchups["T2_avg_Score"] - df_matchups["T1_avg_Score"]
        df_matchups[Columns.QUALITY_DIFF] = df_matchups[Columns.T2_QUALITY] - df_matchups[Columns.T1_QUALITY]
        df_matchups[Columns.ELO_DIFF] = df_matchups[Columns.T1_ELO] - df_matchups[Columns.T2_ELO]
        df_matchups[Columns.ELO_DELTA_DIFF] = df_matchups[Columns.T1_ELO_DELTA] - df_matchups[Columns.T2_ELO_DELTA]
        return df_matchups

    @staticmethod
    def _apply_time_discount(games: pd.DataFrame, discount_factor: float = 0.99) -> pd.DataFrame:
        """
        Apply exponential time-based discounting to game weights.

        More recent games (closer to end of season) receive higher weights.

        Args:
            games (pd.DataFrame): DataFrame with games including Season, DayNum, and GameWeight.

        Returns:
            pd.DataFrame: DataFrame with updated GameWeight including time discount.
        """
        INTER_DAY_NUM = "InterDayNum"
        DAYS_FROM_END = "DaysFromEnd"

        df = games.copy()

        df[Columns.DAY_NUM] = df[Columns.DAY_NUM] - df[Columns.DAY_NUM].min()
        df[INTER_DAY_NUM] = df[Columns.DAY_NUM] + 40 * (df[Columns.SEASON] - df[Columns.SEASON].min())

        df[DAYS_FROM_END] = df[INTER_DAY_NUM].max() - df[INTER_DAY_NUM]
        df[Columns.GAME_WEIGHT] = discount_factor ** df[DAYS_FROM_END]

        return df.drop(columns=[INTER_DAY_NUM, DAYS_FROM_END])

    @staticmethod
    def _compute_weighted_season_stats(games: pd.DataFrame) -> pd.DataFrame:
        """
        Compute weighted average statistics for each team per season.

        Uses GameWeight column to compute weighted averages across all stat columns.
        Formula: weighted_avg = sum(stat * weight) / sum(weight)

        Args:
            games (pd.DataFrame): DataFrame with game data including GameWeight.

        Returns:
            pd.DataFrame: Aggregated team statistics per season.
        """
        exclude_cols = {
            Columns.SEASON,
            Columns.DAY_NUM,
            Columns.T1_TEAM_ID,
            Columns.T2_TEAM_ID,
            Columns.NUM_OT,
            Columns.MEN_WOMEN,
            Columns.TARGET,
        }
        stat_cols = [col for col in games.columns if col not in exclude_cols]

        games_weighted = games.copy()

        for col in stat_cols:
            if pd.api.types.is_numeric_dtype(games_weighted[col]):
                games_weighted[col] = games_weighted[col] * games_weighted[Columns.GAME_WEIGHT]

        weighted_sums = games_weighted.groupby([Columns.T1_TEAM_ID])[stat_cols].sum()

        weight_sums = games.groupby([Columns.T1_TEAM_ID])[Columns.GAME_WEIGHT].sum()

        weighted_averages = weighted_sums.div(weight_sums, axis=0)

        weighted_averages = weighted_averages.reset_index()
        weighted_averages.drop([Columns.GAME_WEIGHT], inplace=True, axis=1)
        return weighted_averages.rename(columns={Columns.T1_TEAM_ID: Columns.TEAM_ID})

    def _split_data(self) -> None:
        """
        Shuffle and split the data into train, validation, and test sets.

        Uses stratification on the target variable to maintain class balance.
        """
        if self._data is None:
            raise RuntimeError("Must call setup() before splitting data")

        # Perform stratified split for train
        train_idx, valid_idx = train_test_split(
            self._data.index,
            train_size=self._train_split,
            test_size=self._valid_split,
            random_state=self._random_seed,
            stratify=self._data[Columns.TARGET],
        )

        self._train_data = self._data.loc[train_idx].reset_index(drop=True)
        self._valid_data = self._data.loc[valid_idx].reset_index(drop=True)
