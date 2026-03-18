"""Canonical rollout Triton entrypoint.

This wrapper keeps the original rollout implementation as the source of truth
so shared imports do not alter the parameter names expected by saved weights.
"""

from experiments.exp2_long_term_stability_test.model.Triton_model import Triton, count_parameters

__all__ = ["Triton", "count_parameters"]

