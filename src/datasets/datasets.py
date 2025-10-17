import pandas as pd

from ..utils.constants import Columns
from ..utils.types import Gender
from .utils import get_data

# Tournament data
TOURNEY_RESULTS = "NCAATourneyCompactResults"
TOURNEY_DETAILED_RESULTS = "NCAATourneyDetailedResults"
SEEDS = "NCAATourneySeeds"
TOURNEY_SLOTS = "NCAATourneySlots"
TOURNEY_SEED_ROUND_SLOTS = "NCAATourneySeedRoundSlots"

# Regular season data
REGULAR_SEASON_RESULTS = "RegularSeasonCompactResults"
REGULAR_SEASON_DETAILED_RESULTS = "RegularSeasonDetailedResults"

# Secondary tournament data
SECONDARY_TOURNEY_RESULTS = "SecondaryTourneyCompactResults"
SECONDARY_TOURNEY_TEAMS = "SecondaryTourneyTeams"

# Conference data
CONFERENCE_TOURNEY_GAMES = "ConferenceTourneyGames"
TEAM_CONFERENCES = "TeamConferences"

# Team data
TEAMS = "Teams"
TEAM_COACHES = "TeamCoaches"
TEAM_SPELLINGS = "TeamSpellings"

# Game location data
GAME_CITIES = "GameCities"

# Ordinals data
MASSEY_ORDINALS = "MasseyOrdinals"

# Seasons data
SEASONS = "Seasons"

# Generic data
CITIES = "Cities"
CONFERENCES = "Conferences"

# Overall ELO ratings
ELO = "OverallCompactResultsWithELO"
ELO_Delta3 = "OverallCompactResultsWithELODelta3_Reset"

# Streaks
STREAKS = "RegularSeasonResultsWithStreaks"

# Quality
QUALITY = "TeamQualitySeasonal"


def seeds() -> pd.DataFrame:
    m_seeds, w_seeds = seeds_per_gender(Gender.MEN), seeds_per_gender(Gender.WOMEN)
    return pd.concat([m_seeds, w_seeds])


def seeds_per_gender(gender: Gender) -> pd.DataFrame:
    seeds = get_data(SEEDS, gender)
    seeds[Columns.REGION] = seeds[Columns.SEED].str[0]
    seeds[Columns.SEED] = seeds[Columns.SEED].str[1:3].astype(int)
    return seeds


def compact_tourney_results() -> pd.DataFrame:
    m_results, w_results = (
        compact_tourney_results_per_gender(Gender.MEN),
        compact_tourney_results_per_gender(Gender.WOMEN),
    )
    return pd.concat([m_results, w_results])


def compact_tourney_results_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(TOURNEY_RESULTS, gender)


def detailed_tourney_results() -> pd.DataFrame:
    m_results, w_results = (
        detailed_tourney_results_per_gender(Gender.MEN),
        detailed_tourney_results_per_gender(Gender.WOMEN),
    )
    return pd.concat([m_results, w_results])


def detailed_tourney_results_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(TOURNEY_DETAILED_RESULTS, gender)


def tourney_slots() -> pd.DataFrame:
    m_slots, w_slots = tourney_slots_per_gender(Gender.MEN), tourney_slots_per_gender(Gender.WOMEN)
    return pd.concat([m_slots, w_slots])


def tourney_slots_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(TOURNEY_SLOTS, gender)


def tourney_seed_round_slots_men() -> pd.DataFrame:
    # Only available for men's tournament
    return get_data(TOURNEY_SEED_ROUND_SLOTS, Gender.MEN)


def compact_regular_season_results() -> pd.DataFrame:
    m_results, w_results = (
        compact_regular_season_results_per_gender(Gender.MEN),
        compact_regular_season_results_per_gender(Gender.WOMEN),
    )
    return pd.concat([m_results, w_results])


def compact_regular_season_results_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(REGULAR_SEASON_RESULTS, gender)


def detailed_regular_season_results() -> pd.DataFrame:
    m_results, w_results = (
        detailed_regular_season_results_per_gender(Gender.MEN),
        detailed_regular_season_results_per_gender(Gender.WOMEN),
    )
    return pd.concat([m_results, w_results])


def detailed_regular_season_results_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(REGULAR_SEASON_DETAILED_RESULTS, gender)


def secondary_tourney_results() -> pd.DataFrame:
    m_results, w_results = (
        secondary_tourney_results_per_gender(Gender.MEN),
        secondary_tourney_results_per_gender(Gender.WOMEN),
    )
    return pd.concat([m_results, w_results])


def secondary_tourney_results_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(SECONDARY_TOURNEY_RESULTS, gender)


def secondary_tourney_teams() -> pd.DataFrame:
    m_teams, w_teams = secondary_tourney_teams_per_gender(Gender.MEN), secondary_tourney_teams_per_gender(Gender.WOMEN)
    return pd.concat([m_teams, w_teams])


def secondary_tourney_teams_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(SECONDARY_TOURNEY_TEAMS, gender)


def conference_tourney_games() -> pd.DataFrame:
    m_games, w_games = (
        conference_tourney_games_per_gender(Gender.MEN),
        conference_tourney_games_per_gender(Gender.WOMEN),
    )
    return pd.concat([m_games, w_games])


def conference_tourney_games_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(CONFERENCE_TOURNEY_GAMES, gender)


def team_conferences() -> pd.DataFrame:
    m_conferences, w_conferences = team_conferences_per_gender(Gender.MEN), team_conferences_per_gender(Gender.WOMEN)
    return pd.concat([m_conferences, w_conferences])


def team_conferences_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(TEAM_CONFERENCES, gender)


def teams() -> pd.DataFrame:
    m_teams, w_teams = teams_per_gender(Gender.MEN), teams_per_gender(Gender.WOMEN)
    return pd.concat([m_teams, w_teams])


def teams_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(TEAMS, gender)


def team_coaches_men() -> pd.DataFrame:
    # Only available for men's teams
    return get_data(TEAM_COACHES, Gender.MEN)


def team_spellings() -> pd.DataFrame:
    m_spellings, w_spellings = team_spellings_per_gender(Gender.MEN), team_spellings_per_gender(Gender.WOMEN)
    return pd.concat([m_spellings, w_spellings])


def team_spellings_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(TEAM_SPELLINGS, gender)


def game_cities() -> pd.DataFrame:
    m_cities, w_cities = game_cities_per_gender(Gender.MEN), game_cities_per_gender(Gender.WOMEN)
    return pd.concat([m_cities, w_cities])


def game_cities_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(GAME_CITIES, gender)


def massey_ordinals_men() -> pd.DataFrame:
    # Only available for men's teams
    return get_data(MASSEY_ORDINALS, Gender.MEN)


def seasons() -> pd.DataFrame:
    m_seasons, w_seasons = seasons_per_gender(Gender.MEN), seasons_per_gender(Gender.WOMEN)
    return pd.concat([m_seasons, w_seasons])


def seasons_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(SEASONS, gender)


def cities() -> pd.DataFrame:
    return get_data(CITIES)


def conferences() -> pd.DataFrame:
    return get_data(CONFERENCES)


def overall_elo_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(ELO, gender)


def overall_elo() -> pd.DataFrame:
    m_elo, w_elo = overall_elo_per_gender(Gender.MEN), overall_elo_per_gender(Gender.WOMEN)
    return pd.concat([m_elo, w_elo])


def overall_elo_delta_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(ELO_Delta3, gender)


def overall_elo_delta() -> pd.DataFrame:
    m_elo_delta, w_elo_delta = overall_elo_delta_per_gender(Gender.MEN), overall_elo_delta_per_gender(Gender.WOMEN)
    return pd.concat([m_elo_delta, w_elo_delta])


def regular_season_streaks_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(STREAKS, gender)


def regular_season_streaks() -> pd.DataFrame:
    m_streaks, w_streaks = (
        regular_season_streaks_per_gender(Gender.MEN),
        regular_season_streaks_per_gender(Gender.WOMEN),
    )
    return pd.concat([m_streaks, w_streaks])


def team_quality_per_gender(gender: Gender) -> pd.DataFrame:
    return get_data(QUALITY, gender)


def team_quality() -> pd.DataFrame:
    m_quality, w_quality = team_quality_per_gender(Gender.MEN), team_quality_per_gender(Gender.WOMEN)
    return pd.concat([m_quality, w_quality])
