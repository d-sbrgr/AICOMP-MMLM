import pandas as pd
from sklearn.model_selection import train_test_split

from src.datasets.datasets import (
    detailed_regular_season_results,
    detailed_tourney_results,
    overall_elo_delta,
    regular_season_streaks,
    seeds,
    team_quality,
)
from src.utils.constants import Columns

from ..base_dataloader import BaseDataloader
from ..common import generate_perspectives, prepare_detailed_results, prepare_engineered_features
from ..feature_selection import FeatureSelection


class WeightedSeasonAvgDataLoader(FeatureSelection, BaseDataloader):
    """
    Loads and prepares weighted rolling season-averaged features for modeling.

    This dataloader creates matchup records from ALL games (regular season + tournament),
    computes weighted aggregations of team stats per season, and generates a dataset
    where each game creates two records (Team A vs Team B, and Team B vs Team A).
    The data is then shuffled and split into train/validation/test sets.

    Key differences from SeasonAverageDataLoader:
    - Uses ALL games (regular + tournament) as training data, not just tournament games
    - Applies time-based discounting to weigh recent games higher
    - Separate weights for regular season vs tournament games
    - Creates bidirectional matchup records
    - Random train/val split instead of temporal split
    """

    def __init__(
        self,
        num_features: int,
        regular_weight: float = 1.0,
        tourney_weight: float = 1.0,
        discount_factor: float = 0.99,
        train_split: float = 0.75,
        valid_split: float = 0.25,
        random_seed: int = 42,
    ):
        """
        Initialize the WeightedSeasonAvgDataLoader.

        Args:
            num_features (int): Number of top features to select for modeling.
            regular_weight (float): Weight multiplier for regular season games (default: 1.0).
            tourney_weight (float): Weight multiplier for tournament games (default: 1.5).
            discount_factor (float): Exponential decay factor for time discounting.
                                    Value closer to 1 = less decay (default: 0.95).
            train_split (float): Fraction of data for training (default: 0.7).
            valid_split (float): Fraction of data for validation (default: 0.15).
                                Remaining data becomes test set.
            random_seed (int): Random seed for reproducible splits (default: 42).
        """
        FeatureSelection.__init__(self, num_features)
        BaseDataloader.__init__(self)
        self._data: pd.DataFrame | None = None
        self._train_data: pd.DataFrame | None = None
        self._valid_data: pd.DataFrame | None = None
        self._df_season_stats_t1: pd.DataFrame | None = None
        self._df_season_stats_t2: pd.DataFrame | None = None

        # Weighting parameters
        self._regular_weight = regular_weight
        self._tourney_weight = tourney_weight
        self._discount_factor = discount_factor

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
        Load, merge, and prepare all regular season and tournament data.

        Process:
        1. Load regular season and tournament detailed results
        2. Merge with ELO and streak features
        3. Apply game type weights (regular vs tournament)
        4. Apply time-based discounting (more recent games weighted higher)
        5. Compute weighted season averages for each team
        6. Generate ALL matchups from both regular season and tournament games
        7. Create bidirectional records (A vs B, B vs A)
        8. Shuffle and split into train/valid/test sets
        """
        # Load and prepare regular season data
        df_regular = prepare_detailed_results(detailed_regular_season_results())
        df_regular[Columns.GAME_WEIGHT] = self._regular_weight

        df_tourney_for_stats = prepare_detailed_results(detailed_tourney_results())
        df_tourney_for_stats[Columns.GAME_WEIGHT] = self._tourney_weight

        elo_delta = prepare_engineered_features(overall_elo_delta())
        streaks = prepare_engineered_features(regular_season_streaks())

        df_regular = pd.merge(
            df_regular,
            elo_delta,
            on=[Columns.SEASON, Columns.DAY_NUM, Columns.T1_TEAM_ID, Columns.T2_TEAM_ID],
            how="left",
        )
        df_regular = pd.merge(
            df_regular,
            streaks,
            on=[Columns.SEASON, Columns.DAY_NUM, Columns.T1_TEAM_ID, Columns.T2_TEAM_ID],
            how="left",
        )

        df_tourney_for_stats = pd.merge(
            df_tourney_for_stats,
            elo_delta,
            on=[Columns.SEASON, Columns.DAY_NUM, Columns.T1_TEAM_ID, Columns.T2_TEAM_ID],
            how="left",
        )
        df_tourney_for_stats = self._add_latest_streak_to_tournaments(df_tourney_for_stats, streaks)

        df_all_games = pd.concat([df_regular, df_tourney_for_stats], ignore_index=True)

        # Apply time-based discount factor
        # For each season, calculate days from end of season and apply exponential decay
        df_all_games = self._apply_time_discount(df_all_games)

        # Compute weighted season statistics for each team
        df_season_stats = self._compute_weighted_season_stats(df_all_games)

        # Add latest ELO to season stats
        df_season_stats = self._add_latest_elo_to_stats(df_all_games, df_season_stats)

        # Add seeds to season stats
        df_season_stats = self._add_seeds_to_stats(seeds(), df_season_stats)

        # Add quality to season stats
        df_season_stats = self._add_quality_to_stats(team_quality(), df_season_stats)

        self._stats = df_season_stats

        # Prepare team stats for merging (T1 and T2 perspectives)
        df_T1_stats, df_T2_stats = generate_perspectives(df_season_stats)
        self._df_season_stats_T1 = df_T1_stats
        self._df_season_stats_T2 = df_T2_stats

        # Generate matchups from ALL games (regular + tournament)
        # This creates the dataset with features from weighted aggregations
        df_matchups = self._generate_matchups(df_all_games)

        # Merge team statistics, seeds, and quality
        self._data = self._merge_stats_with_matchups(df_matchups, df_T1_stats, df_T2_stats)

        # Shuffle and split the data
        self._split_data()

    def _merge_stats_with_matchups(
        self, df_matchups: pd.DataFrame, df_T1_stats: pd.DataFrame, df_T2_stats: pd.DataFrame
    ) -> pd.DataFrame:
        df_matchups = pd.merge(df_matchups, df_T1_stats, on=[Columns.SEASON, Columns.T1_TEAM_ID], how="left")
        df_matchups = pd.merge(df_matchups, df_T2_stats, on=[Columns.SEASON, Columns.T2_TEAM_ID], how="left")

        # Add derived features
        df_matchups[Columns.POINT_DIFF] = df_matchups["T1_avg_Score"] - df_matchups["T2_avg_Score"]
        df_matchups[Columns.SEED_DIFF] = df_matchups[Columns.T2_SEED] - df_matchups[Columns.T1_SEED]
        df_matchups[Columns.QUALITY_DIFF] = df_matchups[Columns.T2_QUALITY] - df_matchups[Columns.T1_QUALITY]
        return df_matchups

    def _add_quality_to_stats(self, df_quality: pd.DataFrame, df_season_stats: pd.DataFrame) -> pd.DataFrame:
        return pd.merge(df_season_stats, df_quality, on=[Columns.SEASON, Columns.TEAM_ID], how="left")

    def _add_seeds_to_stats(self, df_seeds: pd.DataFrame, df_season_stats: pd.DataFrame) -> pd.DataFrame:
        df = pd.merge(df_season_stats, df_seeds, on=[Columns.SEASON, Columns.TEAM_ID], how="left")
        df[Columns.SEED] = df[Columns.SEED].fillna(32)
        return df

    def _add_latest_elo_to_stats(self, df_all_games: pd.DataFrame, df_season_stats: pd.DataFrame) -> pd.DataFrame:
        df_elo = self._get_last_elo_per_season(df_all_games)

        df_season_stats = pd.merge(
            df_season_stats,
            df_elo,
            on=[Columns.SEASON, Columns.TEAM_ID],
            how="left",
        )
        return df_season_stats.rename(columns={Columns.LAST_ELO: "T1_LastElo"})

    def _add_latest_streak_to_tournaments(self, df_tourney: pd.DataFrame, streaks: pd.DataFrame) -> pd.DataFrame:
        """
        Add latest win/loss streaks to tournament games.

        Args:
            df_tourney (pd.DataFrame): Tournament games DataFrame.
            streaks (pd.DataFrame): Streaks DataFrame with T1 and T2 streaks.
        Returns:
            pd.DataFrame: Tournament DataFrame with added streak features.
        """

        streaks = streaks[
            [
                Columns.SEASON,
                Columns.DAY_NUM,
                Columns.T1_TEAM_ID,
                Columns.T2_TEAM_ID,
                Columns.T1_STREAK,
                Columns.T2_STREAK,
            ]
        ]
        streaks_t1 = (
            streaks.sort_values(by=[Columns.SEASON, Columns.DAY_NUM])
            .groupby([Columns.SEASON, Columns.T1_TEAM_ID])
            .agg({Columns.T1_STREAK: "last"})
            .reset_index()
        )
        streaks_t2 = (
            streaks.sort_values(by=[Columns.SEASON, Columns.DAY_NUM])
            .groupby([Columns.SEASON, Columns.T2_TEAM_ID])
            .agg({Columns.T2_STREAK: "last"})
            .reset_index()
        )

        df_tourney = pd.merge(
            df_tourney,
            streaks_t1,
            on=[Columns.SEASON, Columns.T1_TEAM_ID],
            how="left",
        )
        return pd.merge(
            df_tourney,
            streaks_t2,
            on=[Columns.SEASON, Columns.T2_TEAM_ID],
            how="left",
        )

    def _apply_time_discount(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Apply exponential time-based discounting to game weights.

        More recent games (closer to end of season) receive higher weights.

        Args:
            df (pd.DataFrame): DataFrame with games including Season, DayNum, and GameWeight.

        Returns:
            pd.DataFrame: DataFrame with updated GameWeight including time discount.
        """
        if self._discount_factor == 1.0:
            return games

        MAX_DAY_NUM = "MaxDayNum"
        DAYS_FROM_END = "DaysFromEnd"
        TIME_DISCOUNT = "TimeDiscount"

        max_days = games.groupby(Columns.SEASON)[Columns.DAY_NUM].max().reset_index()
        max_days.columns = [Columns.SEASON, MAX_DAY_NUM]
        games = pd.merge(games, max_days, on=Columns.SEASON, how="left")
        games[DAYS_FROM_END] = games[MAX_DAY_NUM] - games[Columns.DAY_NUM]

        # Apply exponential decay: weight * (discount_factor ^ days_from_end)
        games[TIME_DISCOUNT] = self._discount_factor ** games[DAYS_FROM_END]
        games[Columns.GAME_WEIGHT] = games[Columns.GAME_WEIGHT] * games[TIME_DISCOUNT]

        return games.drop(columns=[MAX_DAY_NUM, DAYS_FROM_END, TIME_DISCOUNT])

    def _compute_weighted_season_stats(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Compute weighted average statistics for each team per season.

        Uses GameWeight column to compute weighted averages across all stat columns.
        Formula: weighted_avg = sum(stat * weight) / sum(weight)

        Args:
            df (pd.DataFrame): DataFrame with game data including GameWeight.

        Returns:
            pd.DataFrame: Aggregated team statistics per season.
        """
        # Identify columns to aggregate (exclude identifiers and weights)
        exclude_cols = {
            Columns.SEASON,
            Columns.DAY_NUM,
            Columns.T1_TEAM_ID,
            Columns.T2_TEAM_ID,
            Columns.NUM_OT,
            Columns.MEN_WOMEN,
            Columns.TARGET,
            Columns.GAME_WEIGHT,
        }
        stat_cols = [col for col in games.columns if col not in exclude_cols]

        games_weighted = games.copy()

        for col in stat_cols:
            if pd.api.types.is_numeric_dtype(games_weighted[col]):
                games_weighted[col] = games_weighted[col] * games_weighted[Columns.GAME_WEIGHT]

        weighted_sums = games_weighted.groupby([Columns.SEASON, Columns.T1_TEAM_ID])[stat_cols].sum()

        weight_sums = games.groupby([Columns.SEASON, Columns.T1_TEAM_ID])[Columns.GAME_WEIGHT].sum()

        weighted_averages = weighted_sums.div(weight_sums, axis=0)

        weighted_averages = weighted_averages.reset_index()
        return weighted_averages.rename(columns={Columns.T1_TEAM_ID: Columns.TEAM_ID})

    def _get_last_elo_per_season(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Get the last ELO rating for each team in each season.

        Args:
            df_regular (pd.DataFrame): Regular season data with ELO ratings.

        Returns:
            pd.DataFrame: Last ELO per team per season.
        """
        df_elo: pd.DataFrame = games[[Columns.SEASON, Columns.DAY_NUM, Columns.T1_TEAM_ID, Columns.T1_ELO]].rename(
            columns={Columns.T1_TEAM_ID: Columns.TEAM_ID, Columns.T1_ELO: Columns.LAST_ELO}
        )
        df_elo = (
            df_elo.sort_values([Columns.SEASON, Columns.TEAM_ID, Columns.DAY_NUM])
            .groupby([Columns.SEASON, Columns.TEAM_ID])
            .last()
            .reset_index()
        )
        return df_elo[[Columns.SEASON, Columns.TEAM_ID, Columns.LAST_ELO]]

    def _generate_matchups(self, df_all_games: pd.DataFrame) -> pd.DataFrame:
        """
        Generate matchup records from all games.

        Creates two records per game: (Team1 vs Team2) and (Team2 vs Team1).

        Args:
            df_all_games (pd.DataFrame): All games with features.

        Returns:
            pd.DataFrame: Bidirectional matchup records.
        """
        matchup_cols = [
            Columns.SEASON,
            Columns.T1_TEAM_ID,
            Columns.T2_TEAM_ID,
            Columns.TARGET,
            Columns.MEN_WOMEN,
        ]

        return df_all_games[matchup_cols].copy()

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
            raise RuntimeError("Must call train_data first to set features")
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

        return self._merge_stats_with_matchups(df, self._df_season_stats_T1, self._df_season_stats_T2)
