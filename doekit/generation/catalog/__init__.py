"""Classical and modern design constructors."""

from .factorial import full_factorial, fractional_factorial
from .screening import plackett_burman, is_plackett_burman, fold
from .response_surface import box_behnken, central_composite
from .definitive import definitive_screening
from .random_design import random_design, latin_hypercube
from .mixture import simplex_lattice, simplex_centroid
from .split_plot import split_plot_design
from .registry import DESIGN_BUILDERS, register_design, get_builder

__all__ = [
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
    "DESIGN_BUILDERS",
    "register_design",
    "get_builder",
]
