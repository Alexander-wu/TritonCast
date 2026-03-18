"""Canonical weather and climate Triton entrypoint.

This wrapper intentionally reuses the existing implementation from the
medium-range weather experiment so that model structure and weight loading
remain unchanged while exposing a shared import path.
"""

from experiments.exp1_medium_range_weather_forecasting.models.Triton import Triton, count_parameters

__all__ = ["Triton", "count_parameters"]

