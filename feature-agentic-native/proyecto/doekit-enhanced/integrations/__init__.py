"""
integrations - Adaptadores de integracion externa.
"""

from integrations.bayesian_opt import (
    CandidateScore,
    BayesianOptProposal,
    BayesianOptAdapter,
    propose_with_bayesian_opt,
)


__all__ = [
    "CandidateScore",
    "BayesianOptProposal",
    "BayesianOptAdapter",
    "propose_with_bayesian_opt",
]
