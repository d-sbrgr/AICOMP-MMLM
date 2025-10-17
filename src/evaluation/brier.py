import warnings

import numpy as np
import pandas as pd

from ..utils import get_data_directory, get_submissions_directory
from ..utils.constants import Columns


def _load_submission_data(submission_file: str, year: int) -> pd.DataFrame:
    df_submission = pd.read_csv(get_submissions_directory() / submission_file)

    df_submission[[Columns.SEASON, Columns.LOWER_TEAM, Columns.HIGHER_TEAM]] = df_submission["ID"].str.split(
        "_", expand=True
    )
    df_submission[[Columns.SEASON, Columns.LOWER_TEAM, Columns.HIGHER_TEAM]] = df_submission[
        [Columns.SEASON, Columns.LOWER_TEAM, Columns.HIGHER_TEAM]
    ].apply(pd.to_numeric)

    # ensure only the year in question is kept
    df_submission_year = df_submission[df_submission[Columns.SEASON] == year].copy()

    # return only relevant columns
    return df_submission_year[[Columns.LOWER_TEAM, Columns.HIGHER_TEAM, Columns.PRED]]


def _load_tournament_data(year: int) -> pd.DataFrame:
    cols = [Columns.SEASON, Columns.WTEAM_ID, Columns.LTEAM_ID]
    df_men_tourneys = pd.read_csv(get_data_directory() / "MNCAATourneyCompactResults.csv", usecols=cols)
    df_women_tourneys = pd.read_csv(get_data_directory() / "WNCAATourneyCompactResults.csv", usecols=cols)
    df_tourneys = pd.concat((df_men_tourneys, df_women_tourneys))

    df_tourney_year = df_tourneys[df_tourneys[Columns.SEASON] == year].copy()

    # add column with the 'Result' in terms of the lower ID team -> 1 if lower ID team won, 0 if higher ID team won
    df_tourney_year["Result"] = np.where(df_tourney_year[Columns.WTEAM_ID] < df_tourney_year[Columns.LTEAM_ID], 1, 0)

    # add columns for lowerID and higherID
    df_tourney_year[Columns.LOWER_TEAM] = df_tourney_year[[Columns.WTEAM_ID, Columns.LTEAM_ID]].min(axis=1)
    df_tourney_year[Columns.HIGHER_TEAM] = df_tourney_year[[Columns.WTEAM_ID, Columns.LTEAM_ID]].max(axis=1)

    # return only relevant columns
    return df_tourney_year[[Columns.LOWER_TEAM, Columns.HIGHER_TEAM, Columns.RESULT]]


def _combine_data(df_tourney: pd.DataFrame, df_submission: pd.DataFrame) -> pd.DataFrame:
    # merge the two dataframes on lowerID and higherID
    df_merged = pd.merge(df_tourney, df_submission, on=[Columns.LOWER_TEAM, Columns.HIGHER_TEAM], how="left")
    missing_pred = df_merged[Columns.PRED].isna().sum()

    if missing_pred > 0:
        warnings.warn(f"{missing_pred} rows have no predictions in the merged DataFrame.")
    return df_merged


def brier_score(preds, true) -> float:
    return np.mean((preds - true) ** 2).astype(float)


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
    return compute_brier_score_for_predictions(df_submission, year)


def compute_brier_score_for_predictions(matchup_with_preds: pd.DataFrame, year: int) -> float:
    df_tourney = _load_tournament_data(year)
    df_combined = _combine_data(df_tourney, matchup_with_preds)

    return brier_score(df_combined[Columns.PRED], df_combined[Columns.RESULT])
