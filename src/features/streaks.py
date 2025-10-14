"""
Team Win/Loss Streaks Feature Computation

This module computes win and loss streaks for teams based on game results.
For each game in the dataset, it calculates the current streak of wins or losses
for each team leading up to that game.

WHAT ARE STREAKS?
Streaks represent a team's recent performance momentum:
- Win Streak: Number of consecutive wins before the current game (0 if last game was a loss)
- Loss Streak: Number of consecutive losses before the current game (0 if last game was a win)
- At any point in time, a team either has a win streak OR a loss streak, but not both
- Streaks reset between seasons by default

USAGE EXAMPLES:
- Calculate streaks for regular season data
- Calculate streaks for tournament data
- Calculate streaks for combined regular season + tournament data
- Use streaks as features for prediction models
"""

import pandas as pd
import numpy as np

from typing import Optional
from collections import defaultdict

from ..utils.constants import Columns


def calculate_streaks(
    games: pd.DataFrame,
    reset_between_seasons: bool = True,
    include_current_game: bool = False
) -> pd.DataFrame:
    """
    Calculate win and loss streaks for each team at each point in time.

    For each game, this function determines the win streak and loss streak of each team
    (both winner and loser) leading up to that game. A team can only have either a win
    streak or a loss streak at any given time, never both.

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
        DataFrame with original game data plus four new columns:
            - WWinStreak: Number of consecutive wins for the winning team before this game
            - WLossStreak: Number of consecutive losses for the winning team before this game
            - LWinStreak: Number of consecutive wins for the losing team before this game
            - LLossStreak: Number of consecutive losses for the losing team before this game
    """
    # Validate required columns
    required_cols = {Columns.SEASON, Columns.DAY_NUM, Columns.WTEAM_ID, Columns.LTEAM_ID}
    missing = required_cols - set(games.columns)
    if missing:
        raise ValueError(f"games_df missing required columns: {missing}")

    # Sort by season and day number to ensure chronological order
    sorted_games = games.sort_values([Columns.SEASON, Columns.DAY_NUM]).reset_index(drop=True).copy()

    # Initialize streak tracking dictionaries
    # Each team maps to (current_win_streak, current_loss_streak)
    team_streaks: defaultdict[int, tuple[int, int]] = defaultdict(lambda: (0, 0))

    # Lists to store streak values for each game
    w_win_streaks = []
    w_loss_streaks = []
    l_win_streaks = []
    l_loss_streaks = []

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
        w_wins, w_losses = team_streaks[w_team]
        l_wins, l_losses = team_streaks[l_team]

        team_streaks[w_team] = (w_wins + 1, 0)
        team_streaks[l_team] = (0, l_losses + 1)

        w_relevant_wins, w_relevant_losses = team_streaks[w_team] if include_current_game else (w_wins, w_losses)
        l_relevant_wins, l_relevant_losses = team_streaks[l_team] if include_current_game else (l_wins, l_losses)

        # Store streaks for this game
        w_win_streaks.append(w_relevant_wins)
        w_loss_streaks.append(w_relevant_losses)
        l_win_streaks.append(l_relevant_wins)
        l_loss_streaks.append(l_relevant_losses)

    # Add streak columns to the dataframe
    sorted_games[Columns.WWIN_STREAK] = w_win_streaks
    sorted_games[Columns.WLOSS_STREAK] = w_loss_streaks
    sorted_games[Columns.LWIN_STREAK] = l_win_streaks
    sorted_games[Columns.LLOSS_STREAK] = l_loss_streaks

    return sorted_games
