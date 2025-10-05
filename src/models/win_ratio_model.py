import pandas as pd

from typing import Iterable

from .model import Model
from ..datasets.datasets import compact_tourney_results, compact_regular_season_results
from ..utils.constants import Columns


class WinRatioModel(Model):
    """Predicts the outcome of a matchup based on the overall win rates of the two teams."""

    def __init__(self, alpha: float = 10.0, prior: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.prior = prior
        self.WIN_RATE = "WinRate"
        self.WINS = "Wins"
        self.LOSSES = "Losses"
        self.GAMES = "Games"
        self.WIN_RATE_LOWER = f"{self.WIN_RATE}Lower"
        self.WIN_RATE_HIGHER = f"{self.WIN_RATE}Higher"

    def fit(self, season: int) -> None:
        tourney_results = compact_tourney_results()
        regular_results = compact_regular_season_results()

        # Filter to previous seasons only
        tourney_results = tourney_results[tourney_results[Columns.SEASON] < season]
        regular_results = regular_results[regular_results[Columns.SEASON] <= season]

        tourney_results = tourney_results[[Columns.WTEAM_ID, Columns.LTEAM_ID]]
        regular_results = regular_results[[Columns.WTEAM_ID, Columns.LTEAM_ID]]

        all_results = pd.concat([tourney_results, regular_results])
        self._win_rates = self._calculate_win_rates(all_results)

    def predict(self, matchups: pd.DataFrame) -> Iterable[float]:
        updated_matchups = matchups.merge(self._win_rates, how="left", left_on=Columns.LOWER_TEAM,
                                          right_on=Columns.TEAM_ID).rename(columns={self.WIN_RATE: self.WIN_RATE_LOWER}).drop(columns=[Columns.TEAM_ID])
        updated_matchups = updated_matchups.merge(self._win_rates, how="left", left_on=Columns.HIGHER_TEAM,
                                                  right_on=Columns.TEAM_ID).rename(columns={self.WIN_RATE: self.WIN_RATE_HIGHER}).drop(columns=[Columns.TEAM_ID])

        return updated_matchups[self.WIN_RATE_LOWER] / (updated_matchups[self.WIN_RATE_LOWER] + updated_matchups[self.WIN_RATE_HIGHER])

    def _calculate_win_rates(self, games: pd.DataFrame) -> pd.DataFrame:
        tourney_stats = games.groupby(by=Columns.WTEAM_ID).size().reset_index(name=self.WINS).rename(columns={Columns.WTEAM_ID: Columns.TEAM_ID})
        tourney_stats[self.LOSSES] = games.groupby(by=Columns.LTEAM_ID).size().reset_index(name=self.LOSSES)[self.LOSSES]
        tourney_stats[self.GAMES] = tourney_stats[self.WINS] + tourney_stats[self.LOSSES]

        # Win rate with bayesian smoothing
        tourney_stats[self.WIN_RATE] = (tourney_stats[self.WINS] + self.alpha * self.prior) / (tourney_stats[self.GAMES] + self.alpha)

        return tourney_stats
