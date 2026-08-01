"""Open/Closed registry of design constructors by kind label."""

from __future__ import annotations

from typing import Callable

from .factorial import full_factorial, fractional_factorial
from .screening import plackett_burman
from .response_surface import box_behnken, central_composite
from .definitive import definitive_screening
from .random_design import random_design, latin_hypercube
from .mixture import simplex_lattice, simplex_centroid
from .split_plot import split_plot_design

#: kind metadata tag -> public constructor (or None when not a simple factory)
DESIGN_BUILDERS: dict[str, Callable] = {
    "FullFactorial": full_factorial,
    "FractionalFactorial": fractional_factorial,
    "PlackettBurman": plackett_burman,
    "BoxBehnken": box_behnken,
    "CentralComposite": central_composite,
    "DefinitiveScreening": definitive_screening,
    "RandomDesign": random_design,
    "LatinHypercube": latin_hypercube,
    "SimplexLattice": simplex_lattice,
    "SimplexCentroid": simplex_centroid,
    "SplitPlot": split_plot_design,
}

#: Advisor shortlist labels -> builder callables are registered in advise.rules


def register_design(kind: str, builder: Callable) -> None:
    """Register a design constructor under a ``metadata['kind']`` tag."""
    DESIGN_BUILDERS[kind] = builder


def get_builder(kind: str) -> Callable:
    """Look up a registered design builder."""
    try:
        return DESIGN_BUILDERS[kind]
    except KeyError as exc:
        raise KeyError(f"unknown design kind: {kind!r}. "
                       f"Known: {sorted(DESIGN_BUILDERS)}") from exc
