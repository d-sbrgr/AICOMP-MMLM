import pandas as pd
from typing import Literal
from itertools import combinations

from ..utils import get_data_directory, Gender
from ..utils.constants import Columns


def _load_teams_data(year: int, gender: Gender) -> pd.DataFrame:
    df_teams = pd.read_csv(get_data_directory() / f"{gender.value}RegularSeasonCompactResults.csv")
    df_teams = df_teams[df_teams[Columns.SEASON] == year]
    return pd.DataFrame({Columns.TEAM_ID: sorted(pd.unique(df_teams[[Columns.WTEAM_ID, Columns.LTEAM_ID]].values.ravel()))})


def _create_matchups(df_teams):
    return pd.DataFrame(
        [(min(a, b), max(a, b)) for a, b in combinations(df_teams[Columns.TEAM_ID], 2)],
        columns=[Columns.LOWER_TEAM, Columns.HIGHER_TEAM]
    )


def generate_matchups(year: int) -> pd.DataFrame:
    """
    Generates all possible matchups between teams that participated in Division 1 in the given year.

    Args:
        year (int): The year for which to generate matchups.

    Returns:
        pd.DataFrame: A DataFrame containing all possible matchups with columns 'ID', 'lowerID' and 'higherID'.
    """
    df_men = _create_matchups(_load_teams_data(year, Gender.MEN))
    df_women = _create_matchups(_load_teams_data(year, Gender.WOMEN))
    df_matchups = pd.concat((df_men, df_women)).reset_index(drop=True)
    df_matchups[Columns.ID] = df_matchups.apply(lambda row: f"{year}_{row[Columns.LOWER_TEAM]}_{row[Columns.HIGHER_TEAM]}", axis=1)
    return df_matchups
