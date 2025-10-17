"""
Team Win/Loss Streaks Feature Computation

This module computes win and loss streaks for teams based on game results.
For each game in the dataset, it calculates the current streak of wins (positive) or losses (negative)
for each team leading up to that game.

WHAT ARE STREAKS?
Streaks represent a team's recent performance momentum:
- Win Streak: Number of consecutive wins before the current game (positive integer)
- Loss Streak: Number of consecutive losses before the current game (negative integer)
- At any point in time, a team has a single streak value: positive for wins, negative for losses
- Streaks reset between seasons by default

USAGE EXAMPLES:
- Calculate streaks for regular season data
- Calculate streaks for tournament data
- Calculate streaks for combined regular season + tournament data
- Use streaks as features for prediction models
"""

from collections import defaultdict

import pandas as pd

from ..utils.constants import Columns


def calculate_streaks(
    games: pd.DataFrame, reset_between_seasons: bool = True, include_current_game: bool = False
) -> pd.DataFrame:
    """
    Calculate win/loss streaks for each team at each point in time.

    For each game, this function determines the streak of each team (winner and loser)
    leading up to that game. Streaks are positive for consecutive wins, negative for consecutive losses.

    Args:
        games_df: DataFrame with game results. Must include columns:
            - Season: Season identifier
            - DayNum: Day number within the season
            - WTeamID: Winning team ID
            - LTeamID: Losing team ID
        reset_between_seasons: If True, streaks reset to 0 at the start of each new season.
                               If False, streaks carry over across seasons.
        include_current_game: If True, includes the current game in the streak calculation.
                             If False, calculates the streak up to (but not including) the current game.

    Returns:
        DataFrame with original game data plus two new columns:
            - WStreak: Streak for the winning team before this game (positive for wins, negative for losses)
            - LStreak: Streak for the losing team before this game (positive for wins, negative for losses)
    """
    # Validate required columns
    required_cols = {Columns.SEASON, Columns.DAY_NUM, Columns.WTEAM_ID, Columns.LTEAM_ID}
    missing = required_cols - set(games.columns)
    if missing:
        raise ValueError(f"games_df missing required columns: {missing}")

    # Sort by season and day number to ensure chronological order
    sorted_games = games.sort_values([Columns.SEASON, Columns.DAY_NUM]).reset_index(drop=True).copy()

    # Initialize streak tracking dictionary
    # Each team maps to current streak (positive for wins, negative for losses)
    team_streaks: defaultdict[int, int] = defaultdict(int)

    # Lists to store streak values for each game
    w_streaks = []
    l_streaks = []

    current_season = None

    for _, row in sorted_games.iterrows():
        season = int(row[Columns.SEASON])
        w_team = int(row[Columns.WTEAM_ID])
        l_team = int(row[Columns.LTEAM_ID])

        # Reset streaks at the start of a new season if configured
        if reset_between_seasons and current_season is not None and season != current_season:
            team_streaks.clear()

        current_season = season

        # Get current streaks for both teams (before this game)
        w_streak = team_streaks[w_team]
        l_streak = team_streaks[l_team]

        # Store streaks for this game (before updating for current game)
        w_relevant_streak = team_streaks[w_team] if include_current_game else w_streak
        l_relevant_streak = team_streaks[l_team] if include_current_game else l_streak

        w_streaks.append(w_relevant_streak)
        l_streaks.append(l_relevant_streak)

        # Update streaks for both teams after this game
        # Winner: increment win streak or reset loss streak to 1
        if team_streaks[w_team] > 0:
            team_streaks[w_team] += 1
        else:
            team_streaks[w_team] = 1
        # Loser: increment loss streak (more negative) or reset win streak to -1
        if team_streaks[l_team] < 0:
            team_streaks[l_team] -= 1
        else:
            team_streaks[l_team] = -1

    # Add streak columns to the dataframe
    sorted_games["WStreak"] = w_streaks
    sorted_games["LStreak"] = l_streaks

    return sorted_games
