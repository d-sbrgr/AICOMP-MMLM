from ..models.model import Model
from .brier_score import compute_brier_score_for_predictions
from ..submissions.matchups import generate_matchups
from ..utils.constants import Columns


def evaluate(model: Model, season: int) -> float:
    matchups = generate_matchups(season)
    predictions = model.predict(matchups)
    matchups[Columns.PRED] = predictions
    return compute_brier_score_for_predictions(matchups, year=season)


def fit_and_evaluate(model: Model, season: int) -> float:
    model.fit(season)
    return evaluate(model, season)
