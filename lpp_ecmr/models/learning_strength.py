"""Shared learning-strength composition for the model hierarchy."""

from jax import numpy as jnp
from jaxcmr.typing import Float_

__all__ = ["compose_learning_strength"]


def compose_learning_strength(
    baseline: Float_,
    is_emotional: Float_,
    early_lpp: Float_,
    emotion_scale: Float_,
    lpp_main_scale: Float_,
    lpp_inter_scale: Float_,
) -> Float_:
    """Compose positive categorical and log-linked LPP learning multipliers.

    The ordinary pathway baseline is supplied by the caller: ``phi_i`` for
    temporal-context learning and ``source_learning_rate * phi_i`` for
    source-context learning. ``emotion_scale`` is a multiplier with neutral
    value one: it preserves the ordinary baseline for neutral items and scales
    the baseline for emotional items. The two LPP coefficients are slopes on
    log learning strength. Their neutral value is zero, and exponentiation
    guarantees a positive completed strength without rectification.
    """

    categorical_multiplier = 1.0 + (emotion_scale - 1.0) * is_emotional
    log_lpp_multiplier = (
        lpp_main_scale * early_lpp + lpp_inter_scale * is_emotional * early_lpp
    )
    return baseline * categorical_multiplier * jnp.exp(log_lpp_multiplier)
