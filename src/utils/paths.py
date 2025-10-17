from pathlib import Path


def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).parent.parent.parent.resolve()


def get_data_directory() -> Path:
    """Returns the data directory path."""
    return get_project_root() / "data"


def get_submissions_directory() -> Path:
    """Returns the submissions directory path."""
    return get_project_root() / "submissions"
