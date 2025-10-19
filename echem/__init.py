"""
EChem Analysis Module

This module provides electrochemical spectroscopy analysis capabilities
for tracking spectral changes during CV (Cyclic Voltammetry), 
CA (Chronoamperometry), and CC (Chronocoulometry) experiments.

Key differences from DFS analysis:
- Data structure: Time-series (Time × λ) instead of spatial (H × W × λ)
- Integration: Synchronized with potentiostat data
- Analysis: Cycle-based spectral evolution tracking
"""

from .echem_dataset import EChemDataset
from .echem_analysis import EChemAnalyzer

__all__ = ['EChemDataset', 'EChemAnalyzer']