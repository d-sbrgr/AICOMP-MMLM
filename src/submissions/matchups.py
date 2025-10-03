import pandas as pd
from typing import Literal
from itertools import combinations

from ..utils import get_data_directory, Gender


def _load_teams_data(year: int, gender: Gender) -> pd.DataFrame:
    df_teams = pd.read_csv(get_data_directory() / f"{gender.value}RegularSeasonCompactResults.csv")
    df_teams = df_teams[df_teams['Season'] == year]
    return pd.DataFrame({'TeamID': sorted(pd.unique(df_teams[['WTeamID', 'LTeamID']].values.ravel()))})


def _create_matchups(df_teams):
    return pd.DataFrame(
        [(min(a, b), max(a, b)) for a, b in combinations(df_teams['TeamID'], 2)],
        columns=['lowerID', 'higherID']
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
    df_matchups['ID'] = df_matchups.apply(lambda row: f"{year}_{row['lowerID']}_{row['higherID']}", axis=1)
    return df_matchups
