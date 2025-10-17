import pandas as pd

from ..utils.paths import get_data_directory
from ..utils.types import Gender


def get_data(kind: str, gender: Gender = None) -> pd.DataFrame:
    path = get_data_directory().joinpath(f"{kind}.csv" if gender is None else f"{gender.value}{kind}.csv")
    return pd.read_csv(path)
