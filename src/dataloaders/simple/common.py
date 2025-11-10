import pandas as pd

from src.utils.constants import Columns


def adjust_overtime(games: pd.DataFrame) -> pd.DataFrame:
    adjot = (40 + 5 * games[Columns.NUM_OT]) / 40
    adjcols = list(
        set(games.columns).difference(
            {Columns.DAY_NUM, Columns.LTEAM_ID, Columns.NUM_OT, Columns.SEASON, Columns.WTEAM_ID}
        )
    )
    for col in adjcols:
        games[col] = games[col] / adjot
    return games


def prepare_detailed_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare detailed results by normalizing stats, swapping team perspectives,
    and computing point differentials and targets.

    Args:
        df (pd.DataFrame): Raw detailed results DataFrame.

    Returns:
        pd.DataFrame: Processed results with T1/T2 perspective and targets.
    """
    df = df[list(set(df.columns).difference({Columns.WLOC}))]

    df = adjust_overtime(df)

    output = create_both_perspectives(df)

    output[Columns.POINT_DIFF] = output[Columns.T1_SCORE] - output[Columns.T2_SCORE]
    output[Columns.TARGET] = (output[Columns.POINT_DIFF] > 0) * 1
    output[Columns.MEN_WOMEN] = output[Columns.T1_TEAM_ID].apply(lambda t: int(str(t).startswith("1")))

    return output


def create_both_perspectives(df):
    dfswap = df.copy()
    df.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in list(df.columns)]
    dfswap.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in list(dfswap.columns)]

    return pd.concat([df, dfswap]).reset_index(drop=True)


def prepare_engineered_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and swap additional features (ELO, streaks, quality) for merging.

    Args:
        df (pd.DataFrame): DataFrame with additional features.

    Returns:
        pd.DataFrame: Processed features with difference columns.
    """
    columns = set(games.columns)
    columns &= {
        Columns.SEASON,
        Columns.DAY_NUM,
        Columns.LTEAM_ID,
        Columns.WTEAM_ID,
        Columns.WELO,
        Columns.LELO,
        Columns.WELO_DELTA,
        Columns.LELO_DELTA,
        Columns.W_STREAK,
        Columns.L_STREAK,
    }
    games = games[list(columns)]

    dfswap = games.copy()
    games.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in list(games.columns)]
    dfswap.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in list(dfswap.columns)]

    output = pd.concat([games, dfswap]).reset_index(drop=True)

    # Add difference features
    if any(Columns.ELO in col for col in output.columns):
        output[Columns.ELO_DIFF] = output[Columns.T1_ELO] - output[Columns.T2_ELO]
        output[Columns.ELO_DELTA_DIFF] = output["T1_EloDelta"] - output["T2_EloDelta"]

    if any(Columns.STREAK in col for col in output.columns):
        output[Columns.STREAK_DIFF] = output[Columns.T1_STREAK] - output[Columns.T2_STREAK]

    if any(Columns.QUALITY in col for col in output.columns):
        output[Columns.QUALITY_DIFF] = output[Columns.T1_QUALITY] - output[Columns.T2_QUALITY]

    return output


def generate_perspectives(df_season_stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_season_stats_T1 = df_season_stats.copy()
    df_season_stats_T1.columns = [
        "T1_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(df_season_stats_T1.columns)
    ]
    df_season_stats_T1 = df_season_stats_T1.rename(
        {
            "T1_avg_Season": Columns.SEASON,
            "T1_avg_TeamID": Columns.T1_TEAM_ID,
            "T1_avg_LastElo": Columns.T1_ELO,
            "T1_avg_Quality": Columns.T1_QUALITY,
            "T1_avg_Seed": Columns.T1_SEED,
        },
        axis=1,
    )

    df_season_stats_T2 = df_season_stats.copy()
    df_season_stats_T2.columns = [
        "T2_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(df_season_stats_T2.columns)
    ]
    df_season_stats_T2 = df_season_stats_T2.rename(
        {
            "T2_avg_Season": Columns.SEASON,
            "T2_avg_TeamID": Columns.T2_TEAM_ID,
            "T2_avg_LastElo": Columns.T2_ELO,
            "T2_avg_Quality": Columns.T2_QUALITY,
            "T2_avg_Seed": Columns.T2_SEED,
        },
        axis=1,
    )

    return df_season_stats_T1, df_season_stats_T2
