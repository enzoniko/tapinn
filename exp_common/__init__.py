"""Shared utilities for the NeurIPS experiment suite."""

from .device import get_best_device
from .repro import set_global_seed

__all__ = ["get_best_device", "set_global_seed"]
