"""Decision engine: stop / augment / refine / redesign from experiment signals.

    from doekit.orchestration.decide import decide_next_action, context_from_proposal
    ctx = context_from_proposal(proposal, budget_total=40, budget_spent=18)
    decision = decide_next_action(ctx)      # -> Decision(action, confidence, ...)
"""

from .engine import (
    Decision,
    DecisionContext,
    DecisionScore,
    DecisionPolicy,
    ContinuationScorer,
    ThresholdPolicy,
    RiskAdaptivePolicy,
    BudgetAwarePolicy,
    decide_next_action,
    context_from_proposal,
)

__all__ = [
    "Decision",
    "DecisionContext",
    "DecisionScore",
    "DecisionPolicy",
    "ContinuationScorer",
    "ThresholdPolicy",
    "RiskAdaptivePolicy",
    "BudgetAwarePolicy",
    "decide_next_action",
    "context_from_proposal",
]
