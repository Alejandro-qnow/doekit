"""Statistical analysis: OLS, ANOVA, lack-of-fit, mixed models."""

from .results import FitResult, MixedFitResult
from .ols import fit_linear_model
from .anova import anova_table
from .lof import lack_of_fit
from .mixed import fit_mixed_model
from .effects import main_effects, half_normal_data
from .helpers import attach_blocks
from .ports import EstimatorBackend

__all__ = [
    "FitResult",
    "MixedFitResult",
    "fit_linear_model",
    "anova_table",
    "lack_of_fit",
    "fit_mixed_model",
    "main_effects",
    "half_normal_data",
    "attach_blocks",
    "EstimatorBackend",
]
