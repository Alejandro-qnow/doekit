"""Registry of narrative kind labels / template sets (Open/Closed)."""

from __future__ import annotations

from .i18n import _KIND_TO_LABEL

#: metadata kind -> human label used in report prose
KIND_LABELS: dict[str, str] = dict(_KIND_TO_LABEL)

STANDARD_TEMPLATES = {
    "BoxBehnken", "CentralComposite", "DefinitiveScreening",
    "FullFactorial", "FractionalFactorial", "PlackettBurman",
}
RSM_TEMPLATES = {"BoxBehnken", "CentralComposite", "DefinitiveScreening"}
SCREENING_KINDS = {"PlackettBurman", "FractionalFactorial", "DefinitiveScreening"}


def register_kind_label(kind: str, label: str) -> None:
    """Register a display label for a design ``kind`` tag."""
    KIND_LABELS[kind] = label
    _KIND_TO_LABEL[kind] = label
