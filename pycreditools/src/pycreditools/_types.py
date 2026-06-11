from enum import Enum
from typing import TypedDict


class SimulationMethod(str, Enum):
    ANALYTICAL = "analytical"
    STOCHASTIC = "stochastic"


class ClusteringMethod(str, Enum):
    WARD = "ward"
    IV = "iv"


class Quadrant(str, Enum):
    KEEP_IN = "keep_in"
    SWAP_IN = "swap_in"
    SWAP_OUT = "swap_out"
    KEEP_OUT = "keep_out"


class StageDirection(str, Enum):
    GTE = "gte"
    LTE = "lte"


class PolicySummary(TypedDict):
    """Schema for simulation summary outputs."""

    scenario: str
    applicants: int
    approved: float
    hired: float
    bad_rate: float
