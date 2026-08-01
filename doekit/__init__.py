"""doekit: Design of Experiments (DoE) in Python.

Screening, factorial and response-surface designs, optimal design (D/A/I),
and a design-evaluation layer. Flat API::

    import doekit as ed

    # Screening
    pb = ed.plackett_burman(6)

    # Factorial / real fraction with aliasing
    ff = ed.full_factorial({"A": [-1, 1], "B": [-1, 1]})
    fr = ed.fractional_factorial(3, generators=["C=AB"])

    # Response surface with natural ranges
    bb = ed.box_behnken({"temp": (20, 80), "ph": (3, 9), "conc": (0.1, 0.5)})
    cc = ed.central_composite(3)

    # Optimal design (D/A/I) from a candidate set
    cand = ed.random_design([ed.ContinuousFactor("x1", 0, 1),
                             ed.ContinuousFactor("x2", 0, 1)], n=200)
    opt = ed.optimal_design(cand, n_runs=12, model=ed.Model.parse("0 ~ x1 + x2 + x1:x2"),
                            criterion="D")

    # Analysis
    fit = ed.fit_linear_model(pb, response=y)
"""

__version__ = "0.3.0"

from .factors import (Factor, ContinuousFactor, DiscreteFactor,
                      CategoricalFactor, as_factors, factor_from_dict)
from .model import (Model, Intercept, Main, Interaction, Power)
from . import criteria
from .criteria import (d_criterion, a_criterion, t_criterion, g_criterion,
                       e_criterion, i_criterion)
from .designs import (Design, full_factorial, fractional_factorial,
                      plackett_burman, is_plackett_burman, fold,
                      box_behnken, central_composite, definitive_screening,
                      random_design, latin_hypercube, optimal_design,
                      kl_exchange, fedorov_exchange)
from .analysis import fit_linear_model, main_effects, half_normal_data, FitResult
from .evaluate import (evaluate, efficiencies, power_analysis, vif,
                       alias_matrix, fds_data, DesignEvaluation)
from .report import report_html as report, report_summary
from .recommend import recommend_design, Recommendation

__all__ = [
    # factors
    "Factor", "ContinuousFactor", "DiscreteFactor", "CategoricalFactor", "as_factors",
    "factor_from_dict",
    # model
    "Model", "Intercept", "Main", "Interaction", "Power",
    # criteria
    "criteria", "d_criterion", "a_criterion", "t_criterion", "g_criterion",
    "e_criterion", "i_criterion",
    # designs
    "Design", "full_factorial", "fractional_factorial", "plackett_burman",
    "is_plackett_burman", "fold", "box_behnken", "central_composite",
    "definitive_screening", "random_design", "latin_hypercube",
    "optimal_design", "kl_exchange", "fedorov_exchange",
    # analysis
    "fit_linear_model", "main_effects", "half_normal_data", "FitResult",
    # evaluation / benchmarking
    "evaluate", "efficiencies", "power_analysis", "vif", "alias_matrix",
    "fds_data", "DesignEvaluation",
    # reporting
    "report", "report_summary",
    # design advisor
    "recommend_design", "Recommendation",
]
