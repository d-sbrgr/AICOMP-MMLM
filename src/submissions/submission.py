from pathlib import Path

import pandas as pd

from ..models.model import Model
from ..utils import Columns, get_submissions_directory
from .matchups import generate_matchups


def save_submission(data: pd.DataFrame, filename: str) -> Path:
    """
    Saves the submission DataFrame to a CSV file in the submission directory.

    Args:
        data (pd.DataFrame): DataFrame containing the submission data with columns 'ID' and 'Pred'.
        filename (str): Name of the file to save the submission as.

    Returns:
        Path: Path to the saved submission file.
    """

    if Columns.PRED not in data.columns:
        raise ValueError(f"DataFrame must contain a '{Columns.PRED}' column.")

    if Columns.ID not in data.columns:
        data[Columns.ID] = _build_submission_id(data)

    submission_path = get_submissions_directory() / filename

    submission_data = data[[Columns.ID, Columns.PRED]].copy()
    submission_data.to_csv(submission_path, index=False)

    return submission_path


def _build_submission_id(data: pd.DataFrame) -> pd.Series:
    """
    Builds the 'ID' column for the submission DataFrame based on team IDs and season.

    Args:
        data (pd.DataFrame): DataFrame containing the submission data with team ID columns.

    Returns:
        pd.Series: Series containing the constructed 'ID' values."""

    if Columns.TEAM_A in data.columns and Columns.TEAM_B in data.columns:
        lower_team = data[[Columns.TEAM_A, Columns.TEAM_B]].min(axis=1)
        higher_team = data[[Columns.TEAM_A, Columns.TEAM_B]].max(axis=1)
    elif Columns.LOWER_TEAM in data.columns and Columns.HIGHER_TEAM in data.columns:
        lower_team = data[Columns.LOWER_TEAM]
        higher_team = data[Columns.HIGHER_TEAM]
    else:
        raise ValueError(
            f"DataFrame must contain either '{Columns.TEAM_A}' and '{Columns.TEAM_B}' columns "
            f"or '{Columns.LOWER_TEAM}' and '{Columns.HIGHER_TEAM}' columns."
        )

    return data[Columns.SEASON].astype(str) + "_" + lower_team.astype(str) + "_" + higher_team.astype(str)


def create_submission(season: int, model: Model, filename: str | None = None, fit: bool = True) -> Path:
    """
    Create a submission file for a model and given season and save it in the submission directory.

    Args:
        season (int): The season for which to create the submission.
        model (Model): The (trained) model.
        filename (str): Name of the file to save the submission as.
        fit (bool): Whether to (re)fit the model or not.

    Returns:
        Path: Path to the saved submission file.
    """
    if filename is None:
        filename = f"submission_{str(model).lower()}_season_{season}.csv"

    matchups = generate_matchups(season)
    if fit:
        model.fit(season)
    predictions = model.predict(matchups)
    matchups[Columns.PRED] = predictions

    return save_submission(matchups, filename)
