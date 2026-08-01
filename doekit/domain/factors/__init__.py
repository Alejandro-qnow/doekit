"""Factor abstractions with natural <-> coded conversion."""

from .protocols import Factor
from .continuous import ContinuousFactor
from .discrete import DiscreteFactor
from .categorical import CategoricalFactor
from .mixture import MixtureFactor
from .registry import factor_from_dict, as_factors, register_factor_type
from .coding import encode_frame, decode_frame

__all__ = [
    "Factor",
    "ContinuousFactor",
    "DiscreteFactor",
    "CategoricalFactor",
    "MixtureFactor",
    "factor_from_dict",
    "as_factors",
    "register_factor_type",
    "encode_frame",
    "decode_frame",
]
