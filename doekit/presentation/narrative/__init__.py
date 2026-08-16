"""Pure narrative generation (i18n + rule-based prose)."""

from .i18n import _STRINGS, _t, _KIND_TO_LABEL
from .interpret import Interpretation, interpret

__all__ = ["_STRINGS", "_t", "_KIND_TO_LABEL", "Interpretation", "interpret"]
