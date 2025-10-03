from enum import Enum


class Gender(Enum):
    MEN = "M"
    WOMEN = "W"


class Location(Enum):
    NEUTRAL = "N"
    HOME = "H"
    AWAY = "A"