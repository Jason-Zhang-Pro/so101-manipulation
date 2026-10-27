"""Validate ACT inference cadence shared by evaluation and PPO entry points."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any


_FLOAT32_MAX = 3.4028234663852886e38


def _validate_ensemble_weights(coefficient: float, chunk: int) -> None:
    # LeRobot stores exp(-coefficient * arange(chunk)) and its cumulative
    # sum in float32. A finite Python coefficient can still overflow them.
    if abs(coefficient) > _FLOAT32_MAX:
        raise ValueError("ACT temporal ensemble weights would exceed the finite float32 range.")
    if coefficient >= 0 or chunk == 1:
        return
    growth = -coefficient
    last_exponent = growth * (chunk - 1)
    limit = math.log(_FLOAT32_MAX) - 1e-5
    if last_exponent >= limit:
        raise ValueError("ACT temporal ensemble weights would exceed the finite float32 range.")
    # Stable log of the geometric sum, without allocating a chunk-sized tensor.
    log_weight_sum = (
        last_exponent + math.log(-math.expm1(-growth * chunk))
        - math.log(-math.expm1(-growth))
    )
    if log_weight_sum >= limit:
        raise ValueError("ACT temporal ensemble cumulative weights would exceed the finite float32 range.")


def validate_act_temporal_config(config: Any) -> dict[str, Any]:
    """Validate action timing and return serializable schedule metadata.

    Temporal ensembling predicts a fresh chunk every control step, so its
    executed action count must be one. The chunk horizon is unchanged.
    """
    chunk = getattr(config, "chunk_size", None)
    steps = getattr(config, "n_action_steps", None)
    coefficient = getattr(config, "temporal_ensemble_coeff", None)
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("ACT chunk_size must be a positive integer.")
    if (isinstance(steps, bool) or not isinstance(steps, int)
            or not 1 <= steps <= chunk):
        raise ValueError("ACT n_action_steps must be an integer within [1, chunk_size].")
    if coefficient is not None:
        if (isinstance(coefficient, bool) or not isinstance(coefficient, Real)
                or not math.isfinite(coefficient)):
            raise ValueError("ACT temporal_ensemble_coeff must be a finite real number or None.")
        if steps != 1:
            raise ValueError("ACT temporal ensembling requires n_action_steps=1.")
        coefficient = float(coefficient)
        _validate_ensemble_weights(coefficient, chunk)
    return {
        "chunk_size": chunk,
        "n_action_steps": steps,
        "temporal_ensemble_coeff": coefficient,
        "chunk_schedule": (
            "temporal_ensemble_every_step" if coefficient is not None
            else "fixed_n_action_steps"
        ),
    }
