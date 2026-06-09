from dataclasses import dataclass
from typing import Any, List, Optional, Dict
import pandas as pd

@dataclass
class _CatransInput:
    raw_counts: pd.DataFrame
    metadata: pd.DataFrame
    acute_classification: pd.DataFrame
    model: Any
    batch_col: Optional[str] = None
    model_threshold: float = 0.8
    n_cpus: int = 8
    signature_genes: Optional[List[str]] = None

@dataclass
class _IterationResult:
    iteration: int
    differentially_expressed_genes: int
    differential_expression_analysis_results: pd.DataFrame
    model_classification_results: pd.DataFrame

class CatransResult:
    def __init__(self) -> None:
        self.iterations: List[_IterationResult] = list()
        self.final_classification: Optional[pd.DataFrame] = None
        # self.history: Optional[pd.DataFrame] = None ## ??

    def add_iteration(
            self,
            iteration_index: int,
            de_genes_count: int,
            dea_results: pd.DataFrame,
            model_classification_results: pd.DataFrame
        ) -> None:

        self.iterations.append(
            _IterationResult(
                iteration=iteration_index,
                differentially_expressed_genes=de_genes_count,
                differential_expression_analysis_results=dea_results,
                model_classification_results=model_classification_results
            )
        )

    @property
    def number_of_iterations(self) -> int:
        return len(self.iterations)
    
    @property
    def last_iteration(self) -> _IterationResult:
        return self.iterations[-1]

    def set_final_classification(self, classification: pd.DataFrame) -> None:
        self.final_classification = classification

