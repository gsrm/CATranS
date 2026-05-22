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

class CatransResult:
    def __init__(self) -> None:
        self.iterations: Dict[int, Dict[str, Any]] = {}
        self.final_classification: Optional[pd.DataFrame] = None
        self.history: Optional[pd.DataFrame] = None ## ??

    def add_iteration(self, iteration_index: int, de_genes_count: int, dea_results: pd.DataFrame) -> None:
        self.iterations[iteration_index] = {
            "num_degs": de_genes_count,
            "dea_results": dea_results
        }

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)

    def set_final_classification(self, classification: pd.DataFrame) -> None:
        self.final_classification = classification

