"""
NetworkX-based topology helpers (no array backend).
"""

from .noodle import (
    noodlize_nx,
    split_into_simple_paths_nx,
    connect_matching_paths_nx,
)

__all__ = [
    "noodlize_nx",
    "split_into_simple_paths_nx",
    "connect_matching_paths_nx",
]

