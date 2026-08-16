"""doekit: Design of Experiments (DoE) in Python.

Screening, factorial and response-surface designs, optimal design (D/A/I),
mixture / split-plot, evaluation, analysis, and sequential augmentation::

    import doekit as ed

    pb = ed.plackett_burman(6)
    mix = ed.simplex_lattice(3, degree=2)
    spd = ed.split_plot_design(whole_plot=1, subplot=2)
    nxt = ed.propose_next_runs(pb, response=y, n_add=4)
"""

from ._version import __version__

from .domain.factors import (Factor, ContinuousFactor, DiscreteFactor,
                             CategoricalFactor, MixtureFactor,
                             as_factors, factor_from_dict)
from .domain.model import (Model, Intercept, Main, Interaction, Power)
from .domain import criteria
from .domain.criteria import (d_criterion, a_criterion, t_criterion, g_criterion,
                              e_criterion, i_criterion)
from .domain.design import Design
from .domain.constraints import Constraints, coerce_constraints
from .domain.region import Region, HypercubeRegion, SimplexRegion, region_from_design
from .generation import (full_factorial, fractional_factorial,
                         plackett_burman, is_plackett_burman, fold,
                         box_behnken, central_composite, definitive_screening,
                         random_design, latin_hypercube, optimal_design,
                         kl_exchange, fedorov_exchange,
                         simplex_lattice, simplex_centroid, split_plot_design)
from .assessment.analysis import (fit_linear_model, fit_mixed_model, main_effects,
                                  half_normal_data, anova_table, lack_of_fit,
                                  attach_blocks, FitResult, MixedFitResult)
from .assessment.evaluation import (evaluate, efficiencies, power_analysis, vif,
                                    alias_matrix, fds_data, DesignEvaluation)
from .presentation.report import report_html as report, report_summary
from .presentation.narrative import interpret, Interpretation
from .orchestration.advise import recommend_design, Recommendation
from .orchestration.sequential import (augment_design, propose_next_runs,
                                       compare_designs, NextRunsProposal,
                                       DesignComparison)
from .adapters.bo import candidates_from_bounds, candidates_from_skopt_space
from .assessment.surrogate import (Surrogate, OLSSurrogate, fit_surrogate,
                                   loo_calibration)
from .orchestration.optimize import (expected_improvement,
                                     probability_of_improvement,
                                     upper_confidence_bound,
                                     expected_hypervolume_improvement,
                                     get_acquisition, pareto_front, pareto_mask,
                                     dominates, hypervolume)
from .orchestration.experiment import Experiment, experiment, desirability_scores
from .presentation.export import run_sheet, export_csv, export_excel
from .presentation.workspace import (
    ExperimentProject, Wave, open_project, project, DEFAULT_THRESHOLDS,
    build_conclusions,
)
from .presentation.render import figures_mpl as plotting

__all__ = [
    # factors
    "Factor", "ContinuousFactor", "DiscreteFactor", "CategoricalFactor",
    "MixtureFactor", "as_factors", "factor_from_dict",
    # model
    "Model", "Intercept", "Main", "Interaction", "Power",
    # criteria
    "criteria", "d_criterion", "a_criterion", "t_criterion", "g_criterion",
    "e_criterion", "i_criterion",
    # region / constraints
    "Region", "HypercubeRegion", "SimplexRegion", "region_from_design",
    "Constraints", "coerce_constraints",
    # designs
    "Design", "full_factorial", "fractional_factorial", "plackett_burman",
    "is_plackett_burman", "fold", "box_behnken", "central_composite",
    "definitive_screening", "random_design", "latin_hypercube",
    "optimal_design", "kl_exchange", "fedorov_exchange",
    "simplex_lattice", "simplex_centroid", "split_plot_design",
    # analysis
    "fit_linear_model", "fit_mixed_model", "main_effects", "half_normal_data",
    "anova_table", "lack_of_fit", "attach_blocks", "FitResult", "MixedFitResult",
    # evaluation / benchmarking
    "evaluate", "efficiencies", "power_analysis", "vif", "alias_matrix",
    "fds_data", "DesignEvaluation",
    # reporting
    "report", "report_summary", "interpret", "Interpretation",
    # design advisor
    "recommend_design", "Recommendation",
    # sequential / adaptive
    "augment_design", "propose_next_runs", "compare_designs",
    "NextRunsProposal", "DesignComparison",
    # BO bridge
    "candidates_from_bounds", "candidates_from_skopt_space",
    # surrogate models (optimize intent)
    "Surrogate", "OLSSurrogate", "GPSurrogate", "fit_surrogate",
    "loo_calibration",
    # acquisition / pareto
    "expected_improvement", "probability_of_improvement",
    "upper_confidence_bound", "expected_hypervolume_improvement",
    "get_acquisition", "pareto_front", "pareto_mask", "dominates", "hypervolume",
    # experiment aggregate
    "Experiment", "experiment", "desirability_scores",
    # experiment workspace (traceable project → waves)
    "ExperimentProject", "Wave", "open_project", "project",
    "DEFAULT_THRESHOLDS", "build_conclusions",
    # export
    "run_sheet", "export_csv", "export_excel",
    # plotting (optional matplotlib)
    "plotting",
]


def __getattr__(name):
    # Lazy: GPSurrogate pulls scikit-learn only when actually referenced, so a
    # bare ``import doekit`` never requires the optional ``doekit[bo]`` extra.
    if name == "GPSurrogate":
        from .assessment.surrogate import GPSurrogate  # noqa: PLC0415
        return GPSurrogate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
