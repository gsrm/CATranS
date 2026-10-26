"""
CATranS: Clinical Acute Transcriptomic Stratification.

An iterative, semi-supervised framework for patient stratification from transcriptomic data.
"""

from catrans.core import correlation_based_feature_selection, run_catrans
from catrans.data import CatransClasses, CatransResult

__all__ = [
    'CatransClasses',
    'CatransResult',
    'correlation_based_feature_selection',
    'run_catrans',
    ]