import pandas as pd

from .utils import get_data
from ..utils.types import Gender
from ..utils.constants import Columns

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


def seeds() -> pd.DataFrame:
    m_seeds, w_seeds = seeds_per_gender(Gender.MEN), seeds_per_gender(Gender.WOMEN)
    seeds = pd.concat([m_seeds, w_seeds])
    return seeds


def seeds_per_gender(gender: Gender) -> pd.DataFrame:
    seeds = get_data(SEEDS, gender)
    seeds[Columns.REGION] = seeds[Columns.SEED].str[0]
    seeds[Columns.SEED] = seeds[Columns.SEED].str[1:3].astype(int)
    return seeds


def compact_tourney_results() -> pd.DataFrame:
    m_results, w_results = compact_tourney_results_per_gender(Gender.MEN), compact_tourney_results_per_gender(Gender.WOMEN)
    results = pd.concat([m_results, w_results])
    return results


def compact_tourney_results_per_gender(gender: Gender) -> pd.DataFrame:
    results = get_data(TOURNEY_RESULTS, gender)
    return results


def detailed_tourney_results() -> pd.DataFrame:
    m_results, w_results = detailed_tourney_results_per_gender(Gender.MEN), detailed_tourney_results_per_gender(Gender.WOMEN)
    results = pd.concat([m_results, w_results])
    return results


def detailed_tourney_results_per_gender(gender: Gender) -> pd.DataFrame:
    results = get_data(TOURNEY_DETAILED_RESULTS, gender)
    return results


def tourney_slots() -> pd.DataFrame:
    m_slots, w_slots = tourney_slots_per_gender(Gender.MEN), tourney_slots_per_gender(Gender.WOMEN)
    slots = pd.concat([m_slots, w_slots])
    return slots


def tourney_slots_per_gender(gender: Gender) -> pd.DataFrame:
    slots = get_data(TOURNEY_SLOTS, gender)
    return slots


def tourney_seed_round_slots_men() -> pd.DataFrame:
    # Only available for men's tournament
    slots = get_data(TOURNEY_SEED_ROUND_SLOTS, Gender.MEN)
    return slots


def compact_regular_season_results() -> pd.DataFrame:
    m_results, w_results = compact_regular_season_results_per_gender(Gender.MEN), compact_regular_season_results_per_gender(Gender.WOMEN)
    results = pd.concat([m_results, w_results])
    return results


def compact_regular_season_results_per_gender(gender: Gender) -> pd.DataFrame:
    results = get_data(REGULAR_SEASON_RESULTS, gender)
    return results


def detailed_regular_season_results() -> pd.DataFrame:
    m_results, w_results = detailed_regular_season_results_per_gender(Gender.MEN), detailed_regular_season_results_per_gender(Gender.WOMEN)
    results = pd.concat([m_results, w_results])
    return results


def detailed_regular_season_results_per_gender(gender: Gender) -> pd.DataFrame:
    results = get_data(REGULAR_SEASON_DETAILED_RESULTS, gender)
    return results


def secondary_tourney_results() -> pd.DataFrame:
    m_results, w_results = secondary_tourney_results_per_gender(Gender.MEN), secondary_tourney_results_per_gender(Gender.WOMEN)
    results = pd.concat([m_results, w_results])
    return results


def secondary_tourney_results_per_gender(gender: Gender) -> pd.DataFrame:
    results = get_data(SECONDARY_TOURNEY_RESULTS, gender)
    return results


def secondary_tourney_teams() -> pd.DataFrame:
    m_teams, w_teams = secondary_tourney_teams_per_gender(Gender.MEN), secondary_tourney_teams_per_gender(Gender.WOMEN)
    teams = pd.concat([m_teams, w_teams])
    return teams


def secondary_tourney_teams_per_gender(gender: Gender) -> pd.DataFrame:
    teams = get_data(SECONDARY_TOURNEY_TEAMS, gender)
    return teams


def conference_tourney_games() -> pd.DataFrame:
    m_games, w_games = conference_tourney_games_per_gender(Gender.MEN), conference_tourney_games_per_gender(Gender.WOMEN)
    games = pd.concat([m_games, w_games])
    return games


def conference_tourney_games_per_gender(gender: Gender) -> pd.DataFrame:
    games = get_data(CONFERENCE_TOURNEY_GAMES, gender)
    return games


def team_conferences() -> pd.DataFrame:
    m_conferences, w_conferences = team_conferences_per_gender(Gender.MEN), team_conferences_per_gender(Gender.WOMEN)
    conferences = pd.concat([m_conferences, w_conferences])
    return conferences


def team_conferences_per_gender(gender: Gender) -> pd.DataFrame:
    conferences = get_data(TEAM_CONFERENCES, gender)
    return conferences


def teams() -> pd.DataFrame:
    m_teams, w_teams = teams_per_gender(Gender.MEN), teams_per_gender(Gender.WOMEN)
    teams = pd.concat([m_teams, w_teams])
    return teams


def teams_per_gender(gender: Gender) -> pd.DataFrame:
    teams = get_data(TEAMS, gender)
    return teams


def team_coaches_men() -> pd.DataFrame:
    # Only available for men's teams
    coaches = get_data(TEAM_COACHES, Gender.MEN)
    return coaches


def team_spellings() -> pd.DataFrame:
    m_spellings, w_spellings = team_spellings_per_gender(Gender.MEN), team_spellings_per_gender(Gender.WOMEN)
    spellings = pd.concat([m_spellings, w_spellings])
    return spellings


def team_spellings_per_gender(gender: Gender) -> pd.DataFrame:
    spellings = get_data(TEAM_SPELLINGS, gender)
    return spellings


def game_cities() -> pd.DataFrame:
    m_cities, w_cities = game_cities_per_gender(Gender.MEN), game_cities_per_gender(Gender.WOMEN)
    cities = pd.concat([m_cities, w_cities])
    return cities


def game_cities_per_gender(gender: Gender) -> pd.DataFrame:
    cities = get_data(GAME_CITIES, gender)
    return cities


def massey_ordinals_men() -> pd.DataFrame:
    # Only available for men's teams
    ordinals = get_data(MASSEY_ORDINALS, Gender.MEN)
    return ordinals


def seasons() -> pd.DataFrame:
    m_seasons, w_seasons = seasons_per_gender(Gender.MEN), seasons_per_gender(Gender.WOMEN)
    seasons = pd.concat([m_seasons, w_seasons])
    return seasons


def seasons_per_gender(gender: Gender) -> pd.DataFrame:
    seasons = get_data(SEASONS, gender)
    return seasons


def cities() -> pd.DataFrame:
    return get_data(CITIES)


def conferences() -> pd.DataFrame:
    return get_data(CONFERENCES)
