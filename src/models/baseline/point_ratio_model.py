from collections.abc import Iterable

import pandas as pd

from ...datasets.datasets import compact_regular_season_results, compact_tourney_results
from ...utils.constants import Columns
from ..model import Model

SCORE_LOWER = "ScoreLower"
SCORE_HIGHER = "ScoreHigher"


class PointRatioModel(Model):
    """Predicts the outcome of a matchup based on the ratio of the points scored by the two teams.

    The probability of team A winning against team B is given by:
    P(A beats B) = Points_A / (Points_A + Points_B)

    This results in a higher probability of winning, the more points team A scores compared to team B.
    The scores of each game can be discounted based on how long ago the game was played,
    and weighted based on whether it was a tournament or regular season game.
    """

    def __init__(self, regular_weight: float = 0.5, tourney_weight: float = 1.0, discount_factor: float = 0.9):
        super().__init__()
        self._regular_weight = regular_weight
        self._tourney_weight = tourney_weight
        self._discount_factor = discount_factor

    def fit(self, season: int) -> None:
        tourney_results = compact_tourney_results()
        regular_results = compact_regular_season_results()

        tourney_results = tourney_results[tourney_results[Columns.SEASON] < season]
        regular_results = regular_results[regular_results[Columns.SEASON] <= season]

        rename_rules = {
            Columns.WTEAM_ID: Columns.TEAM_ID,
            Columns.WSCORE: Columns.SCORE,
            Columns.LTEAM_ID: Columns.TEAM_ID,
            Columns.LSCORE: Columns.SCORE,
        }

        tourney_results_winning = tourney_results[[Columns.WTEAM_ID, Columns.WSCORE, Columns.SEASON]].rename(
            columns=rename_rules
        )
        tourney_results_losing = tourney_results[[Columns.LTEAM_ID, Columns.LSCORE, Columns.SEASON]].rename(
            columns=rename_rules
        )
        tourney_results_all = pd.concat([tourney_results_winning, tourney_results_losing])

        regular_results_winning = regular_results[[Columns.WTEAM_ID, Columns.WSCORE, Columns.SEASON]].rename(
            columns=rename_rules
        )
        regular_results_losing = regular_results[[Columns.LTEAM_ID, Columns.LSCORE, Columns.SEASON]].rename(
            columns=rename_rules
        )
        regular_results_all = pd.concat([regular_results_winning, regular_results_losing])

        tourney_results_all[Columns.SCORE] = tourney_results_all[Columns.SCORE] * self._tourney_weight
        regular_results_all[Columns.SCORE] = regular_results_all[Columns.SCORE] * self._regular_weight

        all_results = pd.concat([tourney_results_all, regular_results_all])
        all_results[Columns.SCORE] = all_results[Columns.SCORE] * self._discount_factor ** (
            season - all_results[Columns.SEASON]
        )
        all_results = all_results.drop(columns=[Columns.SEASON])
        self._discounted_points_mean = all_results.groupby(by=Columns.TEAM_ID).mean(numeric_only=True).reset_index()

    def predict(self, matchups: pd.DataFrame) -> Iterable[float]:
        updated_matchups = (
            matchups.merge(
                self._discounted_points_mean, how="left", left_on=Columns.LOWER_TEAM, right_on=Columns.TEAM_ID
            )
            .rename(columns={Columns.SCORE: SCORE_LOWER})
            .drop(columns=[Columns.TEAM_ID])
        )
        updated_matchups = (
            updated_matchups.merge(
                self._discounted_points_mean, how="left", left_on=Columns.HIGHER_TEAM, right_on=Columns.TEAM_ID
            )
            .rename(columns={Columns.SCORE: SCORE_HIGHER})
            .drop(columns=[Columns.TEAM_ID])
        )

        # If no seed is provided for the team, assume that it is a rather bad team
        updated_matchups = updated_matchups.fillna(self._discounted_points_mean[Columns.SCORE].min())

        return updated_matchups[SCORE_LOWER] / (updated_matchups[SCORE_LOWER] + updated_matchups[SCORE_HIGHER])
