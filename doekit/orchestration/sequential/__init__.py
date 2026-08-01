"""Sequential / adaptive DoE orchestration."""

from .propose import (
    augment_design,
    propose_next_runs,
    compare_designs,
    NextRunsProposal,
    DesignComparison,
)

__all__ = [
    "augment_design",
    "propose_next_runs",
    "compare_designs",
    "NextRunsProposal",
    "DesignComparison",
]
