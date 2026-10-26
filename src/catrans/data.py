from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class CatransClasses:
    """
    Container defining the categorical label strings for binary clinical stratification.

    Attributes:
        positive: Label string representing the positive condition (default: "Positive").
            For example, "NonSurvivor", "Relapse", or "Resistant".
        negative: Label string representing the negative / reference condition (default: "Negative").
            For example, "Survivor", "Remission", or "Sensitive".
    """

    positive: str = "Positive"
    negative: str = "Negative"


@dataclass
class _CatransInput:
    """
    Internal container holding validated input datasets and configuration parameters.

    Attributes:
        raw_counts: Raw count matrix of shape (genes x samples).
        metadata: Sample metadata DataFrame.
        acute_classification: DataFrame with 'sample' and 'class' columns.
        model: Supervised classifier instance with predict_proba method.
        classes: CatransClasses defining positive and negative labels.
        batch_col: Metadata column name for technical batch correction, or None.
        model_threshold: Posterior probability cutoff for pseudo-labeling.
        n_cpus: Parallel CPU workers for PyDESeq2.
        signature_genes: Optional list of selected feature gene identifiers.
    """

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
    """
    Container capturing the state and quantitative results of a single relabeling iteration.

    Attributes:
        iteration: Sequential 1-based index of the iteration.
        differentially_expressed_genes: Total count of statistically significant DEGs (padj < 0.05).
        differential_expression_analysis_results: Complete PyDESeq2 differential expression table.
        model_classification_results: DataFrame containing class probability distributions,
            prior labels, and updated classification for all samples in this iteration.
    """

    iteration: int
    differentially_expressed_genes: int
    differential_expression_analysis_results: pd.DataFrame
    model_classification_results: pd.DataFrame


class CatransResult:
    """
    Public results container storing the complete iterative trajectory of the CATranS pipeline.

    Attributes:
        iterations: Chronological list of `_IterationResult` objects for every iteration where
            differential expression signal strictly increased.
        final_classification: DataFrame containing the final converged sample assignments,
            or None if the pipeline has not yet completed.
    """

    def __init__(self) -> None:
        """Initialize an empty CatransResult container."""
        self.iterations: list[_IterationResult] = []
        self.final_classification: pd.DataFrame | None = None

    def _add_iteration(
            self,
            iteration_index: int,
            de_genes_count: int,
            dea_results: pd.DataFrame,
            model_classification_results: pd.DataFrame
        ) -> None:
        """
        Record a successful relabeling iteration snapshot.

        Parameters:
            iteration_index: Sequential 1-based index of this iteration.
            de_genes_count: Number of significant DEGs (padj < 0.05) discovered.
            dea_results: PyDESeq2 differential expression results DataFrame.
            model_classification_results: Classification probabilities and labels DataFrame.
        """
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
        """Return the total number of successful iterations recorded."""
        return len(self.iterations)
    
    @property
    def last_iteration(self) -> _IterationResult:
        """
        Return the `_IterationResult` of the final successful iteration.

        Raises:
            IndexError: If no iterations have been recorded yet.
        """
        return self.iterations[-1]

    def _set_final_classification(self, classification: pd.DataFrame) -> None:
        """
        Store the final converged sample classification DataFrame.

        Parameters:
            classification: DataFrame with sample index and predicted class column.
        """
        self.final_classification = classification

