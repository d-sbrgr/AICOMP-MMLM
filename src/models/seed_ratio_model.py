import pandas as pd

from typing import Iterable

from .model import Model
from ..datasets.datasets import seeds
from ..utils.constants import Columns

MAX_SEED = 16.0
MIN_SEED = 1.0
UNSEEDED = 32.0

SEED_LOWER = "SeedLower"
SEED_HIGHER = "SeedHigher"


class SeedRatioModel(Model):
    """Predicts the outcome of a matchup based on the ratio of the seeds of the two teams.

    The probability of team A winning against team B is given by:
    P(A beats B) = Seed_B / (Seed_A + Seed_B)

    This results in a higher probability of winning, the lower (better) the seed of team A is compared to team B.    
    """

    def __init__(self):
        super().__init__()

    def fit(self, season: int) -> None:
        all_seeds = seeds()
        previous_season = all_seeds[all_seeds[Columns.SEASON] < season]
        previous_season = previous_season.drop(columns=[Columns.SEASON, Columns.REGION])
        seeds_mean = previous_season.groupby(by=Columns.TEAM_ID).mean(numeric_only=True).reset_index()
        self._seeds_mean = seeds_mean

    def predict(self, matchups: pd.DataFrame) -> Iterable[float]:
        updated_matchups = matchups.merge(self._seeds_mean, how="left", left_on=Columns.LOWER_TEAM,
                                          right_on=Columns.TEAM_ID).rename(columns={Columns.SEED: SEED_LOWER}).drop(columns=[Columns.TEAM_ID])
        updated_matchups = updated_matchups.merge(self._seeds_mean, how="left", left_on=Columns.HIGHER_TEAM,
                                                  right_on=Columns.TEAM_ID).rename(columns={Columns.SEED: SEED_HIGHER}).drop(columns=[Columns.TEAM_ID])

        # If no seed is provided for the team, assume that it is a rather bad team
        updated_matchups.fillna(UNSEEDED, inplace=True)

        return updated_matchups[SEED_HIGHER] / (updated_matchups[SEED_LOWER] + updated_matchups[SEED_HIGHER])
