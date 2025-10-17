from collections import defaultdict

import pandas as pd

from ..utils.constants import Columns


def calculate_elo_window_delta(
    games: pd.DataFrame, window_size: int, reset_between_seasons: bool = True
) -> pd.DataFrame:
    """
    Calculate Elo delta for each team based on a rolling window of previous games.

    The delta represents the change in Elo over the last N games for each team.
    For each game, it calculates: current_elo - elo_from_N_games_ago

    Args:
        games: DataFrame with game results containing WElo and LElo columns
        window_size: Number of previous games to look back for delta calculation
        reset_between_seasons: If True, reset window tracking between seasons

    Returns:
        DataFrame with added columns: WEloDelta and LEloDelta
    """
    required_cols = {Columns.SEASON, Columns.DAY_NUM, Columns.WTEAM_ID, Columns.LTEAM_ID, Columns.WELO, Columns.LELO}
    missing = required_cols - set(games.columns)
    if missing:
        raise ValueError(f"games missing required columns: {missing}")

    sorted_games = games.sort_values([Columns.SEASON, Columns.DAY_NUM]).reset_index(drop=True).copy()

    team_elo_history: defaultdict[int, list[float]] = defaultdict(list)

    w_elo_deltas = []
    l_elo_deltas = []

    current_season = None

    for _, row in sorted_games.iterrows():
        season = int(row[Columns.SEASON])
        w_team = int(row[Columns.WTEAM_ID])
        l_team = int(row[Columns.LTEAM_ID])
        w_elo = float(row[Columns.WELO])
        l_elo = float(row[Columns.LELO])

        if reset_between_seasons and current_season is not None and season != current_season:
            team_elo_history.clear()

        current_season = season

        w_history = team_elo_history[w_team]
        l_history = team_elo_history[l_team]

        w_delta = w_elo - w_history[-window_size] if len(w_history) >= window_size else 0.0
        l_delta = l_elo - l_history[-window_size] if len(l_history) >= window_size else 0.0

        w_elo_deltas.append(w_delta)
        l_elo_deltas.append(l_delta)

        team_elo_history[w_team].append(w_elo)
        team_elo_history[l_team].append(l_elo)

    sorted_games[Columns.WELO_DELTA] = w_elo_deltas
    sorted_games[Columns.LELO_DELTA] = l_elo_deltas

    return sorted_games
