"""Default SoundSpaces backend for OCC Data Synth.

The legacy geometric pipeline now lives under ``legacy_geometric`` and is kept
only for comparison and fallback experiments.
"""

from .backend import SoundSpacesBackend, SoundSpacesUnavailableError, check_soundspaces_available
from .config import SoundSpacesConfig
from .coordinate import habitat_to_occ, occ_to_habitat

__all__ = [
    "SoundSpacesBackend",
    "SoundSpacesConfig",
    "SoundSpacesUnavailableError",
    "check_soundspaces_available",
    "habitat_to_occ",
    "occ_to_habitat",
]
