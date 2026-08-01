"""doekit: Design of Experiments (DoE) in Python.

Screening, factorial and response-surface designs, optimal design (D/A/I),
design evaluation, analysis, and sequential augmentation::

    import doekit as ed

    pb = ed.plackett_burman(6)
    fit = ed.fit_linear_model(pb, response=y)

    # Sequential: propose the next batch
    nxt = ed.propose_next_runs(pb, response=y, n_add=4)
    print(nxt.comparison.summary)
"""

__version__ = "0.5.0"

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
from .analysis import (fit_linear_model, fit_mixed_model, main_effects,
                       half_normal_data, anova_table, lack_of_fit,
                       attach_blocks, FitResult, MixedFitResult)
from .evaluate import (evaluate, efficiencies, power_analysis, vif,
                       alias_matrix, fds_data, DesignEvaluation)
from .report import report_html as report, report_summary
from .recommend import recommend_design, Recommendation
from .sequential import (augment_design, propose_next_runs, compare_designs,
                         NextRunsProposal, DesignComparison)
from .bo import candidates_from_bounds, candidates_from_skopt_space

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
    "fit_linear_model", "fit_mixed_model", "main_effects", "half_normal_data",
    "anova_table", "lack_of_fit", "attach_blocks", "FitResult", "MixedFitResult",
    # evaluation / benchmarking
    "evaluate", "efficiencies", "power_analysis", "vif", "alias_matrix",
    "fds_data", "DesignEvaluation",
    # reporting
    "report", "report_summary",
    # design advisor
    "recommend_design", "Recommendation",
    # sequential / adaptive
    "augment_design", "propose_next_runs", "compare_designs",
    "NextRunsProposal", "DesignComparison",
    # BO bridge
    "candidates_from_bounds", "candidates_from_skopt_space",
]
