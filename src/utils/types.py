from enum import Enum


class Gender(Enum):
    MEN = "M"
    WOMEN = "W"
    BOTH = ""


class Location(Enum):
    NEUTRAL = "N"
    HOME = "H"
    AWAY = "A"