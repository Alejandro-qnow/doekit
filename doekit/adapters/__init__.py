"""External integration adapters (skopt, future export/CLI/MCP consumers)."""

from .bo import candidates_from_bounds, candidates_from_skopt_space

__all__ = ["candidates_from_bounds", "candidates_from_skopt_space"]
