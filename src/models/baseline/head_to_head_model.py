from collections.abc import Iterable

import pandas as pd

from ...datasets.datasets import compact_regular_season_results, compact_tourney_results
from ...utils.constants import Columns
from .win_ratio_model import WinRatioModel

WINS_LOWER = "WinsLower"
WINS_HIGHER = "WinsHigher"
TOTAL_GAMES_LOWER = "TotalGamesLower"
TOTAL_GAMES_HIGHER = "TotalGamesHigher"

COUNT = "Count"
WINNING = "Winning"
LOSING = "Losing"
WINNING_WTEAMID = f"WTeamID{WINNING}"
WINNING_LTEAMID = f"LTeamID{WINNING}"
LOSING_WTEAMID = f"WTeamID{LOSING}"
LOSING_LTEAMID = f"LTeamID{LOSING}"
OPPONENT = f"Opponent{Columns.TEAM_ID}"
H2H_WIN_RATE = "H2HWinRate"


class HeadToHeadModel(WinRatioModel):
    """Predicts the outcome of a matchup based on the head-to-head win rates of the two teams.

    If not enough head-to-head data is available, it falls back to using overall win rates.
    """

    def __init__(self, alpha: float = 10.0, prior: float = 0.5, min_head_to_head_games: int = 3):
        super().__init__(alpha=alpha, prior=prior)
        self.min_head_to_head_games = min_head_to_head_games

    def fit(self, season: int) -> None:
        tourney_results = compact_tourney_results()
        regular_results = compact_regular_season_results()

        # Filter to previous seasons only
        tourney_results = tourney_results[tourney_results[Columns.SEASON] < season]
        regular_results = regular_results[regular_results[Columns.SEASON] < season]

        tourney_results = tourney_results[[Columns.WTEAM_ID, Columns.LTEAM_ID]]
        regular_results = regular_results[[Columns.WTEAM_ID, Columns.LTEAM_ID]]

        all_results = pd.concat([tourney_results, regular_results])

        self._overall_win_rates = self._calculate_win_rates(all_results)
        self._head_to_head_win_rates = self._calculate_head_to_head_win_rate(all_results)

    def predict(self, matchups: pd.DataFrame) -> Iterable[float]:
        updated_matchups = matchups.merge(
            self._head_to_head_win_rates,
            how="left",
            left_on=[Columns.LOWER_TEAM, Columns.HIGHER_TEAM],
            right_on=[Columns.TEAM_ID, OPPONENT],
        ).drop(columns=[Columns.TEAM_ID, OPPONENT])

        return updated_matchups[H2H_WIN_RATE].fillna(0.5)

    def _calculate_head_to_head_win_rate(self, games: pd.DataFrame) -> pd.DataFrame:
        game_wins = games.groupby(by=[Columns.WTEAM_ID, Columns.LTEAM_ID]).size().reset_index(name=self.WINS)
        game_losses = games.groupby(by=[Columns.LTEAM_ID, Columns.WTEAM_ID]).size().reset_index(name=self.LOSSES)

        head_to_head_games = game_wins.merge(
            game_losses,
            left_on=[Columns.WTEAM_ID, Columns.LTEAM_ID],
            right_on=[Columns.LTEAM_ID, Columns.WTEAM_ID],
            how="outer",
            suffixes=(WINNING, LOSING),
        )

        head_to_head_games[WINNING_WTEAMID] = head_to_head_games[WINNING_WTEAMID].fillna(
            head_to_head_games[LOSING_LTEAMID]
        )
        head_to_head_games[WINNING_LTEAMID] = head_to_head_games[WINNING_LTEAMID].fillna(
            head_to_head_games[LOSING_WTEAMID]
        )
        head_to_head_games[LOSING_WTEAMID] = head_to_head_games[LOSING_WTEAMID].fillna(
            head_to_head_games[WINNING_LTEAMID]
        )
        head_to_head_games[LOSING_LTEAMID] = head_to_head_games[LOSING_LTEAMID].fillna(
            head_to_head_games[WINNING_WTEAMID]
        )
        head_to_head_games[[self.WINS, self.LOSSES]] = head_to_head_games[[self.WINS, self.LOSSES]].fillna(0)

        head_to_head_games = head_to_head_games.drop(columns=[LOSING_WTEAMID, LOSING_LTEAMID])
        head_to_head_games = head_to_head_games.rename(
            columns={WINNING_WTEAMID: Columns.TEAM_ID, WINNING_LTEAMID: OPPONENT}
        )

        head_to_head_games[COUNT] = head_to_head_games[self.WINS] + head_to_head_games[self.LOSSES]

        # Merge overall win rates for both teams to use as fallback
        head_to_head_games = head_to_head_games.merge(
            self._overall_win_rates[[Columns.TEAM_ID, self.WIN_RATE]],
            on=Columns.TEAM_ID,
            how="left",
            suffixes=("", "_Team"),
        )
        head_to_head_games = head_to_head_games.merge(
            self._overall_win_rates[[Columns.TEAM_ID, self.WIN_RATE]],
            left_on=OPPONENT,
            right_on=Columns.TEAM_ID,
            how="left",
            suffixes=("", "_Opponent"),
        ).drop(columns=[f"{Columns.TEAM_ID}_Opponent"])

        # Direct head to head win rate with bayesian smoothing
        head_to_head_games[H2H_WIN_RATE] = (head_to_head_games[self.WINS] + self.alpha * self.prior) / (
            head_to_head_games[self.WINS] + head_to_head_games[self.LOSSES] + self.alpha
        )

        # Win rate ratio
        overall_ratio = head_to_head_games[f"{self.WIN_RATE}"] / (
            head_to_head_games[f"{self.WIN_RATE}"] + head_to_head_games[f"{self.WIN_RATE}_Opponent"]
        )

        # Use head-to-head when sufficient games, otherwise use overall ratio
        head_to_head_games[H2H_WIN_RATE] = head_to_head_games[H2H_WIN_RATE].where(
            head_to_head_games[COUNT] >= self.min_head_to_head_games, other=overall_ratio
        )

        return head_to_head_games.drop(columns=[f"{self.WIN_RATE}", f"{self.WIN_RATE}_Opponent"])
