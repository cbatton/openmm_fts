# omm/__init__.py
"""OpenMM interfaces for various simulation methods."""

from .omm_fts import OMMFF
from .omm_replica import OMMFFReplica

__all__ = ["OMMFF", "OMMFFReplica"]
