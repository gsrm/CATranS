from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class CatransClasses:
    positive: str = "Positive"
    negative: str = "Negative"

@dataclass
class _CatransInput:
    raw_counts: pd.DataFrame
    metadata: pd.DataFrame
    acute_classification: pd.DataFrame
    model: Any
    classes: CatransClasses
    batch_col: str | None = None
    model_threshold: float = 0.8
    n_cpus: int = 8
    signature_genes: list[str] | None = None

@dataclass
class _IterationResult:
    iteration: int
    differentially_expressed_genes: int
    differential_expression_analysis_results: pd.DataFrame
    model_classification_results: pd.DataFrame

class CatransResult:
    def __init__(self) -> None:
        self.iterations: list[_IterationResult] = []
        self.final_classification: pd.DataFrame | None = None

    def _add_iteration(
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

    def _set_final_classification(self, classification: pd.DataFrame) -> None:      
        self.final_classification = classification

