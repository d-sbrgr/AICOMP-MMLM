"""
Team Quality Feature Computation

This module computes team quality ratings using linear regression (GLM) on regular season game data.
The quality metric represents each team's expected point contribution in a matchup.

WHAT IS QUALITY?
Quality is a statistical rating (in points) that estimates team strength based on game results.
- Higher quality = stronger team (tends to outscore opponents)
- Quality difference ≈ expected point margin in a matchup
- Uses GLM (Generalized Linear Model) with Gaussian family
- Formula: PointDiff ~ -1 + T1_TeamID + T2_TeamID
- No intercept (-1) because point diff should be zero for equal teams
- Separate models for men's and women's basketball
"""

import pandas as pd
import statsmodels.api as sm

from ..datasets.datasets import compact_regular_season_results_per_gender
from ..utils.constants import Columns
from ..utils.types import Gender

# Column name prefixes for team 1 and team 2
T1 = "T1_"
T2 = "T2_"

# Score column names
SCORE = "Score"
T1_SCORE = f"{T1}{SCORE}"
T2_SCORE = f"{T2}{SCORE}"

# Team ID column names
TEAM_ID = "TeamID"
T1_TEAM_ID = f"{T1}{TEAM_ID}"
T2_TEAM_ID = f"{T2}{TEAM_ID}"

# Basketball game timing constants
REGULAR_DURATION = 40  # Regular game duration in minutes
OVERTIME_DURATION = 5  # Each overtime period in minutes


def apply_quality_to_matchups(matchups: pd.DataFrame, team_columns: list[str], quality: pd.DataFrame) -> pd.DataFrame:
    """
    Apply team quality ratings to a set of matchups.
    Args:
        matchups: DataFrame with matchups to enhance. Must include team ID columns specified in team_columns.
        team_columns: List of two column names in matchups representing the two teams (e.g., ["WTeamID", "LTeamID"]).
        quality: DataFrame with team quality ratings. Must include Columns.TEAM_ID and Columns.QUALITY,
            and optionally Columns.SEASON.
    Returns:
        DataFrame with original matchups plus:
        - Quality rating for team in team_columns[0]
        - Quality rating for team in team_columns[1]
        - Difference in quality ratings (team_columns[0] - team_columns[1])
    """
    assert len(team_columns) == 2, "team_columns must have exactly two team ID column names"

    matchups_with_quality = matchups.copy()
    team_a, team_b = team_columns
    team_a_quality_col, team_b_quality_col = _team_quality_name(team_a), _team_quality_name(team_b)

    seasonal = Columns.SEASON in matchups.columns and Columns.SEASON in quality.columns

    for team_col in team_columns:
        matchups_with_quality = (
            matchups_with_quality.merge(
                quality,
                how="left",
                left_on=[team_col, Columns.SEASON] if seasonal else team_col,
                right_on=[Columns.TEAM_ID, Columns.SEASON] if seasonal else Columns.TEAM_ID,
            )
            .rename(columns={Columns.QUALITY: _team_quality_name(team_col)})
            .drop(columns=[Columns.TEAM_ID])
        )

    matchups_with_quality[Columns.QUALITY_DIFF] = (
        matchups_with_quality[team_a_quality_col] - matchups_with_quality[team_b_quality_col]
    )
    return matchups_with_quality


def compute_quality(
    quality_base_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute team quality ratings using linear regression (GLM).

    What is "quality"?
    - A number (in points) representing how strong a team is
    - Positive quality = team tends to win by more points
    - Negative quality = team tends to lose by more points
    - Quality difference between two teams ≈ expected point margin

    How it works:
    The function fits a statistical model that says:
        "Point difference in a game = Team1's quality - Team2's quality + randomness"

    For example:
    - If Duke has quality +10 and UNC has quality +5
    - When Duke plays UNC, we expect Duke to win by ~5 points (10 - 5 = 5)

    Technical details:
    Uses a Generalized Linear Model (GLM) with the formula:
        {Columns.POINT_DIFF} ~ -1 + {T1}{Columns.TEAM_ID} + {T2}{Columns.TEAM_ID}
    This estimates each team's contribution to the point differential.

    Args:
        quality_base_data: Prepared game data from get_quality_base_data()
                          Must have columns: {T1}{Columns.TEAM_ID}, {T2}{Columns.TEAM_ID}, {Columns.POINT_DIFF}

    Returns:
        DataFrame with columns:
        - {Columns.TEAM_ID}: Team identifier
        - {Columns.QUALITY}: Team's quality rating (in points)

        Returns empty DataFrame if:
        - No games in the input data
        - GLM fitting fails (e.g., too few teams, numerical issues)
    """
    empty_result = pd.DataFrame(columns=[Columns.TEAM_ID, Columns.QUALITY])
    if quality_base_data.shape[0] == 0:
        return empty_result

    quality_base_data[T1_TEAM_ID] = quality_base_data[T1_TEAM_ID].astype(str)
    quality_base_data[T2_TEAM_ID] = quality_base_data[T2_TEAM_ID].astype(str)

    formula = f"{Columns.POINT_DIFF} ~ -1 + {T1_TEAM_ID} + {T2_TEAM_ID}"

    try:
        glm = sm.GLM.from_formula(
            formula=formula,
            data=quality_base_data,
            family=sm.families.Gaussian(),
        ).fit()
    except Exception:
        print("Warning: GLM fit failed")
        return empty_result

    # Extract coefficients
    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = [Columns.TEAM_ID, Columns.QUALITY]

    # Filter to T1_ coefficients (positive team effects)
    quality = quality.loc[quality[Columns.TEAM_ID].str.contains(T1)].reset_index(drop=True)

    # Extract numeric team ID from parameter name
    # Parameter names are like "T1_TeamID[T.1234]" -> extract "1234"
    quality[Columns.TEAM_ID] = (
        quality[Columns.TEAM_ID].apply(lambda x: x.split("[T.")[1].rstrip("]") if "[T." in x else x[10:14]).astype(int)
    )

    return quality


def compute_team_quality_per_season(
    gender: Gender = Gender.BOTH, min_season: int = 0, max_season: int = 9999
) -> pd.DataFrame:
    """
    Compute team quality ratings for multiple seasons.

    This is the main function to use! It computes quality ratings for all teams
    across all seasons in the specified range.

    Why per season?
    - Teams change every year (new players, different roster strength)
    - Quality is calculated separately for each season
    - A team might be quality +15 in 2023 but only +8 in 2024

    Args:
        gender: Which teams to analyze

        min_season: Earliest season to include (default: 0 = all available)
                    Example: min_season=2020 starts from 2020 season

        max_season: Latest season to include (default: 9999 = all available)
                    Example: max_season=2024 stops at 2024 season

    Returns:
        DataFrame with columns:
        - {Columns.TEAM_ID}: Team identifier
        - {Columns.QUALITY}: Team's quality rating for that season (in points)
        - {Columns.SEASON}: Year of the season
    """
    if gender == Gender.BOTH:
        w_quality = compute_team_quality_per_season(Gender.WOMEN, min_season, max_season)
        m_quality = compute_team_quality_per_season(Gender.MEN, min_season, max_season)
        return pd.concat([w_quality, m_quality]).reset_index(drop=True)

    data = _get_quality_base_data(gender)
    seasons = data[Columns.SEASON].unique().tolist()
    seasons = [s for s in seasons if min_season <= s <= max_season]

    qualities = []
    for season in seasons:
        print(f"Computing quality for season {season} ({gender})...")
        season_data = data[data[Columns.SEASON] == season].copy()
        season_quality = compute_quality(season_data)
        season_quality[Columns.SEASON] = season
        qualities.append(season_quality)

    return pd.concat(qualities).reset_index(drop=True)


def _get_quality_base_data(gender: Gender) -> pd.DataFrame:
    """
    Load and prepare regular season game data for quality computation.

    This function:
    1. Loads all regular season games for the specified gender
    2. Adjusts scores for overtime (normalizes to 40-minute games)
    3. Creates a "doubled" dataset where each game appears twice with teams swapped
       - Game 1: Team A vs Team B (actual result)
       - Game 2: Team B vs Team A (swapped positions, same outcome)

    Introducing the redundancy makes the linear regression symmetric - it learns
    that a team's strength is the same whether they're listed as T1 or T2.

    Args:
        gender: Gender.MEN or Gender.WOMEN (which dataset to load)

    Returns:
        DataFrame with columns:
        - {Columns.SEASON}: Year of the season
        - {T1}{Columns.TEAM_ID}: First team's ID
        - {T2}{Columns.TEAM_ID}: Second team's ID
        - {T1}{Columns.SCORE}: First team's score (adjusted for OT)
        - {T2}{Columns.SCORE}: Second team's score (adjusted for OT)
        - {Columns.POINT_DIFF}: {T1}{Columns.SCORE} - {T2}{Columns.SCORE}
    """
    regular_match_data = compact_regular_season_results_per_gender(gender)

    # Select only columns needed for quality computation
    regular_match_data = regular_match_data[
        [Columns.SEASON, Columns.LTEAM_ID, Columns.LSCORE, Columns.WTEAM_ID, Columns.WSCORE, Columns.NUM_OT]
    ].copy()

    # Adjust for overtimes (normalize to 40-minute game)
    adjot = (REGULAR_DURATION + OVERTIME_DURATION * regular_match_data[Columns.NUM_OT]) / REGULAR_DURATION
    regular_match_data[Columns.WSCORE] = regular_match_data[Columns.WSCORE] / adjot
    regular_match_data[Columns.LSCORE] = regular_match_data[Columns.LSCORE] / adjot

    swapped_season_data = regular_match_data.copy()
    regular_match_data.columns = [x.replace("W", T1).replace("L", T2) for x in regular_match_data.columns]
    swapped_season_data.columns = [x.replace("L", T1).replace("W", T2) for x in swapped_season_data.columns]

    output = pd.concat([regular_match_data, swapped_season_data]).reset_index(drop=True)
    output[Columns.POINT_DIFF] = output[T1_SCORE] - output[T2_SCORE]
    return output


def _team_quality_name(team_col: str) -> str:
    return f"{team_col}{Columns.QUALITY}"
