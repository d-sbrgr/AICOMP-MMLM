import pandas as pd
import numpy as np
import warnings

from ..utils import get_submissions_directory, get_data_directory


def _load_submission_data(submission_file: str, year: int) -> pd.DataFrame:
    df_submission = pd.read_csv(get_submissions_directory() / submission_file)

    df_submission[['Season', 'lowerID', 'higherID']] = df_submission['ID'].str.split('_', expand=True)
    df_submission[['Season', 'lowerID', 'higherID']] = df_submission[['Season', 'lowerID', 'higherID']].apply(
        pd.to_numeric)

    # ensure only the year in question is kept
    df_submission_year = df_submission[df_submission['Season'] == year].copy()

    # return only relevant columns
    return df_submission_year[['lowerID', 'higherID', 'Pred']]


def _load_tournament_data(year: int) -> pd.DataFrame:
    cols = ['Season', 'WTeamID', 'LTeamID']
    df_men_tourneys = pd.read_csv(get_data_directory() / "MNCAATourneyCompactResults.csv", usecols=cols)
    df_women_tourneys = pd.read_csv(get_data_directory() / "WNCAATourneyCompactResults.csv", usecols=cols)
    df_tourneys = pd.concat((df_men_tourneys, df_women_tourneys))

    df_tourney_year = df_tourneys[df_tourneys['Season'] == year].copy()

    # add column with the 'Result' in terms of the lower ID team -> 1 if lower ID team won, 0 if higher ID team won
    df_tourney_year['Result'] = np.where(df_tourney_year['WTeamID'] < df_tourney_year['LTeamID'], 1, 0)

    # add columns for lowerID and higherID
    df_tourney_year['lowerID'] = df_tourney_year[['WTeamID', 'LTeamID']].min(axis=1)
    df_tourney_year['higherID'] = df_tourney_year[['WTeamID', 'LTeamID']].max(axis=1)

    # return only relevant columns
    return df_tourney_year[['lowerID', 'higherID', 'Result']]


def _combine_data(df_tourney: pd.DataFrame, df_submission: pd.DataFrame) -> pd.DataFrame:
    # merge the two dataframes on lowerID and higherID
    df_merged = pd.merge(df_tourney, df_submission, on=['lowerID', 'higherID'], how='left')
    missing_pred = df_merged['Pred'].isna().sum()

    if missing_pred > 0:
        warnings.warn(f"{missing_pred} rows have no predictions in the merged DataFrame.")
    return df_merged


def compute_brier_score(submission_file: str, year: int) -> float:
    """
    Computes the Brier score for a given submission file and year.

    Args:
        submission_file (str): Name of the submissions file in the submissions directory.
        year (int): The tournament year for which to compute the Brier score.

    Returns:
        float: The computed Brier score.
    """
    df_submission = _load_submission_data(submission_file, year)
    df_tourney = _load_tournament_data(year)
    df_combined = _combine_data(df_tourney, df_submission)

    return np.mean((df_combined['Pred'] - df_combined['Result']) ** 2).astype(float)
