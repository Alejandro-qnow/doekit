"""Model specification DSL and model-matrix construction."""

from .terms import Intercept, Main, Interaction, Power, Term, term_from_dict
from .spec import Model

__all__ = [
    "Intercept",
    "Main",
    "Interaction",
    "Power",
    "Term",
    "term_from_dict",
    "Model",
]
