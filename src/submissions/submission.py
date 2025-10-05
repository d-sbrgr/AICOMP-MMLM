import pandas as pd
from ..utils import get_submissions_directory
from ..utils import Columns
from ..models.model import Model


def save_submission(data: pd.DataFrame, filename: str) -> str:
    """
    Saves the submission DataFrame to a CSV file in the submissions directory.

    Args:
        data (pd.DataFrame): DataFrame containing the submission data with columns 'ID' and 'Pred'.
        filename (str): Name of the file to save the submission as.

    Returns:
        str: Path to the saved submission file.
    """

    if Columns.PRED not in data.columns:
        raise ValueError(f"DataFrame must contain a '{Columns.PRED}' column.")

    if Columns.ID not in data.columns:
        data[Columns.ID] = _build_submission_id(data)

    submission_path = get_submissions_directory() / filename

    submission_data = data[[Columns.ID, Columns.PRED]].copy()
    submission_data.to_csv(submission_path, index=False)

    return submission_path, submission_data


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
            f"DataFrame must contain either '{Columns.TEAM_A}' and '{Columns.TEAM_B}' columns or '{Columns.LOWER_TEAM}' and '{Columns.HIGHER_TEAM}' columns."
        )

    return data[Columns.SEASON].astype(str) + "_" + lower_team.astype(str) + "_" + higher_team.astype(str)


def create_submission(season: int, model: Model, filename: str = None, fit: bool = True) -> str:
    from .matchups import generate_matchups

    if filename is None:
        filename = f"submission_{str(model).lower()}_season_{season}.csv"

    matchups = generate_matchups(season)
    if fit:
        model.fit(season)
    predictions = model.predict(matchups)
    matchups[Columns.PRED] = predictions

    return save_submission(matchups, filename)
