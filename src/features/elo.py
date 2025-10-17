from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..datasets.datasets import (
    compact_regular_season_results,
    compact_regular_season_results_per_gender,
    compact_tourney_results,
    compact_tourney_results_per_gender,
)
from ..utils import Columns, Gender


@dataclass
class EloConfig:
    """
    Configuration for Elo rating calculations.

    Attributes:
        base (float): Baseline Elo rating for new teams (default: 1000.0)
        K (float): Update factor determining sensitivity to game outcomes (default: 20.0)
        carry (float): Fraction of previous season's deviation from base to carry over (range: 0 to 1, default: 0.75)
        use_margin (bool): Whether to adjust K based on margin of victory (default: True)
    """

    base: float = 1000.0
    K: float = 20.0
    carry: float = 0.75
    use_margin: bool = True


@dataclass
class RunConfig:
    """
    Configuration of datasets and seasons to include in Elo calculation.

    Attributes:
        gender (Gender): Gender of the teams to include
        max_regular_season (int | None): Maximum season year to include (inclusive). If None, includes all seasons.
        min_regular_season (int | None): Minimum season year to include (inclusive). If None, includes all seasons.
    """

    gender: Gender
    max_regular_season: int | None = None
    min_regular_season: int | None = None


def _initialize_elos(teams: Iterable[int], base: float) -> dict[int, float]:
    return {int(t): base for t in teams}


def _season_carryover(prev_elos: dict[int, float], cfg: EloConfig) -> dict[int, float]:
    next_elos = {}
    for tid, elo in prev_elos.items():
        next_elos[tid] = (cfg.carry * elo) + (1.0 - cfg.carry) * cfg.base
    return next_elos


def calculate_elo(
    rcfg: RunConfig, ecfg: EloConfig, include_predictions: bool = False
) -> tuple[pd.DataFrame, dict[int, float]]:
    """ "
    Calculate Elo ratings for teams based on game results.
    The Elo ratings are calculated based on regular season and tournament results only.

    Args:
        rcfg (RunConfig): Configuration for data selection, defining gender and which seasons to include
        ecfg (EloConfig): Configuration for Elo calculation parameters
        include_predictions (bool): Whether to include predicted win probabilities in the output DataFrame

    Returns:
        Tuple containing:
            - DataFrame with game results and corresponding Elo ratings (and predictions if requested)
            - Dictionary mapping team IDs to their final Elo ratings
    """
    if rcfg.gender == Gender.BOTH:
        df_regular = compact_regular_season_results()
        df_tourney = compact_tourney_results()
    else:
        df_regular = compact_regular_season_results_per_gender(rcfg.gender)
        df_tourney = compact_tourney_results_per_gender(rcfg.gender)

    if rcfg.min_regular_season is not None:
        df_regular = df_regular[df_regular["Season"] >= rcfg.min_regular_season]
        df_tourney = df_tourney[df_tourney["Season"] >= rcfg.min_regular_season]
    if rcfg.max_regular_season is not None:
        df_regular = df_regular[df_regular["Season"] <= rcfg.max_regular_season]
        df_tourney = df_tourney[df_tourney["Season"] < rcfg.max_regular_season]
    else:
        df_tourney = df_tourney[df_tourney["Season"] < df_regular["Season"].max()]

    df = pd.concat([df_regular, df_tourney], ignore_index=True)

    req_cols = {Columns.SEASON, Columns.DAY_NUM, Columns.WTEAM_ID, Columns.LTEAM_ID}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {missing}")

    df_elo = df.sort_values([Columns.SEASON, Columns.DAY_NUM]).reset_index(drop=True).copy()

    teams = pd.Index(np.unique(df_elo[[Columns.WTEAM_ID, Columns.LTEAM_ID]].values)).astype(int).tolist()
    elos: dict[int, float] = _initialize_elos(teams, ecfg.base)

    seasons = df_elo[Columns.SEASON].unique().tolist()
    preds = []
    w_elos = []
    l_elos = []

    cur_season = seasons[0]
    for _, row in df_elo.iterrows():
        season = int(row.Season)
        if season != cur_season:
            elos = _season_carryover(elos, ecfg)
            cur_season = season

        winning_team, losing_team = int(row.WTeamID), int(row.LTeamID)

        w_elo, l_elo = elos.get(winning_team, ecfg.base), elos.get(losing_team, ecfg.base)
        p_win = predict_win(w_elo, l_elo)
        preds.append(p_win)
        w_elos.append(w_elo)
        l_elos.append(l_elo)

        k = ecfg.K
        if ecfg.use_margin and {Columns.WSCORE, Columns.LSCORE}.issubset(df_elo.columns):
            margin = max(1, int(row.WScore) - int(row.LScore))
            mov_mult = np.log(margin + 1.0)
            k = ecfg.K * mov_mult

        elos[winning_team] = w_elo + k * (1.0 - p_win)
        elos[losing_team] = l_elo + k * (0.0 - (1.0 - p_win))

    df_elo[Columns.WELO] = w_elos
    df_elo[Columns.LELO] = l_elos
    if include_predictions:
        df_elo[Columns.PRED] = preds
    return df_elo, elos


def predict_win(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
