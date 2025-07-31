"""OpenMM FTS package: examples of advanced methods applied to alanine dipeptide."""

__version__ = "0.1.0"

from .io import TrajWriter
from .omm.omm_fts import OMMFF
from .omm.omm_replica import OMMFFReplica

__all__ = [
    "OMMFF",
    "OMMFFReplica",
    "TrajWriter",
]
