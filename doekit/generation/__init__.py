"""Design generation: catalog constructors and exchange search."""

from .catalog import (
    full_factorial,
    fractional_factorial,
    plackett_burman,
    is_plackett_burman,
    fold,
    box_behnken,
    central_composite,
    definitive_screening,
    random_design,
    latin_hypercube,
    simplex_lattice,
    simplex_centroid,
    split_plot_design,
)
from .search import optimal_design, kl_exchange, fedorov_exchange
from ..domain.design import Design

__all__ = [
    "Design",
    "full_factorial",
    "fractional_factorial",
    "plackett_burman",
    "is_plackett_burman",
    "fold",
    "box_behnken",
    "central_composite",
    "definitive_screening",
    "random_design",
    "latin_hypercube",
    "simplex_lattice",
    "simplex_centroid",
    "split_plot_design",
    "optimal_design",
    "kl_exchange",
    "fedorov_exchange",
]
