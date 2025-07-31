"""OpenMM FTS package: examples of advanced methods applied to alanine dipeptide."""

__version__ = "0.1.0"

from .io import TrajWriter
from .omm import OMMFF, OMMFFReplica

__all__ = [
    "OMMFF",
    "OMMFFReplica",
    "TrajWriter",
]
