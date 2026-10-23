import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from catrans._logging import CustomFormatter
from catrans.core import (
    _diffExpAnalysis,
    _get_normalized_dataset,
    correlation_based_feature_selection,
    run_catrans,
)
from catrans.data import CatransClasses, CatransResult


def test_catrans_result_data_structure():
    """Verify that the CatransResult container correctly stores iterations and classifications."""
    result = CatransResult()
    assert result.number_of_iterations == 0
    assert result.final_classification is None
    
    # Add a mock iteration
    mock_df = pd.DataFrame({"stat": [1, 2]})
    mock_class_results = pd.DataFrame(
        {
            "negative_class_probability": [0.9, 0.1],
            "positive_class_probability": [0.1, 0.9],
            "prior_classification": ["Alive", "Dead"],
            "current_classification": ["Alive", "Dead"],
        },
        index=["Sample1", "Sample2"]  # ty:ignore[invalid-argument-type]
    )
    result._add_iteration(
        iteration_index=1,
        de_genes_count=10,
        dea_results=mock_df,
        model_classification_results=mock_class_results
    )
    assert result.number_of_iterations == 1
    assert result.iterations[0].differentially_expressed_genes == 10
    assert result.last_iteration.differentially_expressed_genes == 10
    
    # Set final classification
    final_df = pd.DataFrame({"class": ["Alive", "Dead"]})
    result._set_final_classification(final_df)
    assert result.final_classification is not None
    assert result.final_classification.iloc[0]["class"] == "Alive"

def test_normalization_without_batch_correction():
    """Verify that _get_normalized_dataset performs median-of-ratios scale-normalization."""
    # 5 genes, 4 samples
    counts = pd.DataFrame(
        [
            [10, 20, 15, 30],
            [5, 10, 8, 12],
            [100, 200, 150, 280],
            [2, 4, 3, 5],
            [0, 1, 0, 2]
        ],
        index=[f"Gene{i}" for i in range(5)], # ty:ignore[invalid-argument-type]
        columns=[f"Sample{i}" for i in range(4)] # ty:ignore[invalid-argument-type]
    )
    metadata = pd.DataFrame({
        "sample": [f"Sample{i}" for i in range(4)],
        "condition": ["A", "A", "B", "B"]
    }, index=[f"Sample{i}" for i in range(4)]) # ty:ignore[invalid-argument-type]
    
    # Test normalization with log transform
    norm_counts = _get_normalized_dataset(counts, metadata, log_transform=True)
    assert norm_counts.shape == counts.shape
    # Check that they are log-scaled (non-negative, small values)
    assert (norm_counts >= 0).all().all()
    assert norm_counts.max().max() < 10.0

    # Test normalization without log transform
    raw_norm_counts = _get_normalized_dataset(counts, metadata, log_transform=False)
    assert raw_norm_counts.shape == counts.shape
    assert (raw_norm_counts >= 0).all().all()


def test_correlation_based_feature_selection():
    """Verify feature selection correctly identifies biomarkers and collapses co-expressed clusters."""
    # Construct a synthetic count dataset where Gene0 and Gene1 are identical, and correlate with classification
    # Gene2 is noise.
    np.random.seed(42)
    genes = [f"Gene_{i}" for i in range(3)]
    samples = [f"Sample_{j}" for j in range(8)]
    
    # Gene0 and Gene1 perfectly correlate with Alive/Dead states
    # Let's say: Samples 0-3 are "Alive" (values ~5), Samples 4-7 are "Dead" (values ~50)
    gene0_counts = [5, 6, 5, 4, 50, 48, 52, 50]
    gene1_counts = [4, 5, 5, 5, 49, 50, 51, 48]
    gene2_counts = [10, 40, 2, 80, 5, 12, 90, 1]  # Noise
    
    counts = pd.DataFrame(
        [gene0_counts, gene1_counts, gene2_counts],
        index=genes, # ty:ignore[invalid-argument-type]
        columns=samples # ty:ignore[invalid-argument-type]
    )
    
    acute_classification = pd.DataFrame({
        "sample": samples,
        "class": ["Alive"] * 4 + ["Dead"] * 4
    }, index=samples) # ty:ignore[invalid-argument-type]
    
    # We expect Gene0 and Gene1 to be highly correlated with label and co-expressed.
    # Therefore, graph connected components should group them and select only ONE representative gene,
    # while Gene2 is ignored or discarded due to lack of label correlation or p-value filtering.
    classes = CatransClasses(positive="Dead", negative="Alive")
    selected = correlation_based_feature_selection(
        data=counts,
        acute_classification=acute_classification,
        classes=classes,
        pval=0.05,
        cutoff_corr=0.6,
        cutoff_corr_genes=0.8
    )
    
    assert selected is not None
    # We should have exactly 1 gene selected (either Gene_0 or Gene_1, which was grouped into the same cluster)
    assert len(selected) == 1
    assert selected[0] in ["Gene_0", "Gene_1"]


def test_run_catrans_end_to_end():
    """Verify the entire run_catrans pipeline operates correctly using a fitted sklearn estimator."""
    np.random.seed(42)
    genes = [f"Gene_{i}" for i in range(5)]
    samples = [f"Sample_{j}" for j in range(12)]
    
    # Construct a dataset with real signal in Gene_0 and Gene_1
    # Alive samples: Sample_0,1,2 (initial) and Sample_6,7,8 (unlabeled) -> value ~ 5
    # Dead samples: Sample_3,4,5 (initial) and Sample_9,10,11 (unlabeled) -> value ~ 500
    gene0_counts = [5, 6, 5, 500, 480, 520, 5, 6, 5, 500, 480, 520]
    gene1_counts = [4, 5, 5, 490, 500, 510, 4, 5, 5, 490, 500, 510]
    gene2_counts = [10, 40, 2, 80, 5, 12, 90, 1, 15, 35, 8, 70]
    gene3_counts = [15, 20, 8, 30, 2, 11, 45, 3, 25, 40, 12, 60]
    gene4_counts = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    
    raw_counts = pd.DataFrame(
        [gene0_counts, gene1_counts, gene2_counts, gene3_counts, gene4_counts],
        index=genes, # ty:ignore[invalid-argument-type]
        columns=samples # ty:ignore[invalid-argument-type]
    )
    
    metadata = pd.DataFrame({
        "sample": samples,
        "batch": ["A"] * 6 + ["B"] * 6
    }, index=samples) # ty:ignore[invalid-argument-type]
    
    acute_classification = pd.DataFrame({
        "sample": samples,
        "class": ["Alive"] * 3 + ["Dead"] * 3 + [np.nan] * 6
    }, index=samples) # ty:ignore[invalid-argument-type]
    
    # Initialize simple classifier
    model = LogisticRegression(random_state=42)
    
    # Execute the CATranS pipeline
    # Provide predefined signature genes to bypass feature selection for speed/determinism
    signature = ["Gene_0", "Gene_1", "Gene_2"]
    
    classes = CatransClasses(positive="Dead", negative="Alive")
    result = run_catrans(
        raw_counts=raw_counts,
        metadata=metadata,
        acute_classification=acute_classification,
        model=model,
        classes=classes,
        batch_col="batch",
        model_threshold=0.5,
        n_cpus=1,
        signature_genes=signature,
        verbose=True
    )
    
    assert result is not None
    assert result.final_classification is not None
    assert result.final_classification.shape[0] == 12

def test_run_catrans_custom_classes():
    """Verify run_catrans works correctly with user-defined custom classes."""
    np.random.seed(42)
    genes = [f"Gene_{i}" for i in range(5)]
    samples = [f"Sample_{j}" for j in range(12)]
    
    # Construct a dataset with real signal in Gene_0 and Gene_1
    # Mild samples: Sample_0,1,2 (initial) and Sample_6,7,8 (unlabeled) -> value ~ 5
    # Severe samples: Sample_3,4,5 (initial) and Sample_9,10,11 (unlabeled) -> value ~ 500
    gene0_counts = [5, 6, 5, 500, 480, 520, 5, 6, 5, 500, 480, 520]
    gene1_counts = [4, 5, 5, 490, 500, 510, 4, 5, 5, 490, 500, 510]
    gene2_counts = [10, 40, 2, 80, 5, 12, 90, 1, 15, 35, 8, 70]
    gene3_counts = [15, 20, 8, 30, 2, 11, 45, 3, 25, 40, 12, 60]
    gene4_counts = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    
    raw_counts = pd.DataFrame(
        [gene0_counts, gene1_counts, gene2_counts, gene3_counts, gene4_counts],
        index=genes, # ty:ignore[invalid-argument-type]
        columns=samples # ty:ignore[invalid-argument-type]
    )
    
    metadata = pd.DataFrame({
        "sample": samples,
        "batch": ["A"] * 6 + ["B"] * 6
    }, index=samples) # ty:ignore[invalid-argument-type]
    
    # Use custom labels: "Severe" (positive) and "Mild" (negative)
    classes = CatransClasses(positive="Severe", negative="Mild")
    
    acute_classification = pd.DataFrame({
        "sample": samples,
        "class": ["Mild"] * 3 + ["Severe"] * 3 + [np.nan] * 6
    }, index=samples) # ty:ignore[invalid-argument-type]
    
    model = LogisticRegression(random_state=42)
    signature = ["Gene_0", "Gene_1", "Gene_2"]
    
    result = run_catrans(
        raw_counts=raw_counts,
        metadata=metadata,
        acute_classification=acute_classification,
        model=model,
        classes=classes,
        batch_col="batch",
        model_threshold=0.75,
        n_cpus=1,
        signature_genes=signature,
        verbose=True
    )
    
    assert result is not None
    assert result.final_classification is not None
    assert result.final_classification.shape[0] == 12
    # Ensure we mapped back to custom classes, not hardcoded ones
    final_classes = set(result.final_classification["class"].dropna().unique())
    assert final_classes.issubset({"Severe", "Mild"})


def test_run_catrans_input_validation():
    """Verify run_catrans validates input arguments and raises descriptive exceptions."""
    counts = pd.DataFrame(np.ones((3, 4)), index=["G1", "G2", "G3"], columns=[f"S{i}" for i in range(4)])
    meta = pd.DataFrame({"sample": [f"S{i}" for i in range(4)]})
    model = LogisticRegression()

    # 1. acute_classification is not a DataFrame
    with pytest.raises(TypeError, match=r"must be a pandas\.DataFrame"):
        run_catrans(counts, meta, acute_classification="not_a_df", model=model)  # type: ignore[arg-type]

    # 2. Missing 'sample' or 'class' column
    with pytest.raises(ValueError, match="'sample' and 'class' columns"):
        run_catrans(counts, meta, pd.DataFrame({"wrong": [1, 2]}), model=model)

    # 3. Invalid class labels not in CatransClasses
    invalid_labels_df = pd.DataFrame({"sample": ["S0", "S1"], "class": ["Unknown1", "Unknown2"]})
    with pytest.raises(ValueError, match="Found invalid labels"):
        run_catrans(counts, meta, invalid_labels_df, model=model)

    # 4. Model without predict_proba method
    class ModelWithoutProba:
        pass
    valid_df = pd.DataFrame({"sample": ["S0", "S1"], "class": ["Positive", "Negative"]})
    with pytest.raises(AttributeError, match="predict_proba"):
        run_catrans(counts, meta, valid_df, model=ModelWithoutProba())


def test_run_catrans_default_feature_selection():
    """Verify run_catrans works with default signature_genes=None and default classes=None."""
    np.random.seed(42)
    genes = [f"Gene_{i}" for i in range(5)]
    samples = [f"Sample_{j}" for j in range(12)]
    
    gene0_counts = [5, 6, 5, 500, 480, 520, 5, 6, 5, 500, 480, 520]
    gene1_counts = [4, 5, 5, 490, 500, 510, 4, 5, 5, 490, 500, 510]
    gene2_counts = [10, 40, 2, 80, 5, 12, 90, 1, 15, 35, 8, 70]
    gene3_counts = [15, 20, 8, 30, 2, 11, 45, 3, 25, 40, 12, 60]
    gene4_counts = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    
    raw_counts = pd.DataFrame(
        [gene0_counts, gene1_counts, gene2_counts, gene3_counts, gene4_counts],
        index=genes,
        columns=samples
    )
    metadata = pd.DataFrame({"sample": samples}, index=samples)
    acute_classification = pd.DataFrame({
        "sample": samples,
        "class": ["Negative"] * 3 + ["Positive"] * 3 + [np.nan] * 6
    }, index=samples)
    
    model = LogisticRegression(random_state=42)
    # signature_genes=None triggers automatic feature selection; classes=None defaults to Positive/Negative
    result = run_catrans(
        raw_counts=raw_counts,
        metadata=metadata,
        acute_classification=acute_classification,
        model=model,
        classes=None,
        signature_genes=None,
        n_cpus=1
    )
    assert result is not None
    assert result.final_classification is not None


def test_correlation_based_feature_selection_default_cutoffs_and_fallback():
    """Verify feature selection default Q3 cutoffs and the fallback branch when no genes exceed cutoff."""
    np.random.seed(42)
    genes = [f"Gene_{i}" for i in range(4)]
    samples = [f"Sample_{j}" for j in range(8)]
    
    gene0 = [5, 6, 5, 4, 50, 48, 52, 50]
    gene1 = [4, 5, 5, 5, 49, 50, 51, 48]
    gene2 = [5, 4, 6, 5, 51, 49, 48, 52]
    gene3 = [10, 40, 2, 80, 5, 12, 90, 1]
    
    counts = pd.DataFrame([gene0, gene1, gene2, gene3], index=genes, columns=samples)
    acute_classification = pd.DataFrame({
        "sample": samples,
        "class": ["Alive"] * 4 + ["Dead"] * 4
    }, index=samples)
    classes = CatransClasses(positive="Dead", negative="Alive")
    
    # 1. Test automatic Q3 cutoff computation (cutoff_corr=None, cutoff_corr_genes=None)
    selected = correlation_based_feature_selection(
        data=counts,
        acute_classification=acute_classification,
        classes=classes,
        pval=0.05,
        cutoff_corr=None,
        cutoff_corr_genes=None,
        verbose=True
    )
    assert selected is not None
    assert len(selected) > 0

    # 2. Test fallback branch when cutoff_corr is set too high (no genes exceed cutoff)
    selected_fallback = correlation_based_feature_selection(
        data=counts,
        acute_classification=acute_classification,
        classes=classes,
        pval=0.05,
        cutoff_corr=0.999,
        cutoff_corr_genes=0.8
    )
    assert selected_fallback is not None


def test_diffexp_analysis_edge_cases():
    """Verify _diffExpAnalysis handles fewer than 2 classes and invalid sample matching."""
    counts = pd.DataFrame(
        [[10, 20, 15, 30], [5, 10, 8, 12]],
        index=["Gene0", "Gene1"],
        columns=["S0", "S1", "S2", "S3"]
    )
    
    # 1. Fewer than 2 unique classes present
    single_class_df = pd.DataFrame({
        "sample": ["S0", "S1", "S2", "S3"],
        "class": ["Positive"] * 4
    })
    degs, dea = _diffExpAnalysis(counts, single_class_df, CatransClasses(), n_cpus=1)
    assert degs == 0
    assert dea.empty

    # 2. Samples in classification not found in count matrix
    mismatched_df = pd.DataFrame({
        "sample": ["NonExistent1", "NonExistent2"],
        "class": ["Positive", "Negative"]
    })
    with pytest.raises(ValueError, match="Classification must be a dataframe"):
        _diffExpAnalysis(counts, mismatched_df, CatransClasses(), n_cpus=1)


def test_logging_custom_formatter_debug():
    """Verify CustomFormatter switches to debug_format when record is DEBUG."""
    formatter = CustomFormatter()
    record_debug = logging.LogRecord(
        name="test_logger",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=10,
        msg="debug message",
        args=(),
        exc_info=None,
        func="test_func"
    )
    formatted = formatter.format(record_debug)
    assert "STEP: test_func" in formatted


def test_correlation_based_feature_selection_debug_logging():
    """Verify correlation_based_feature_selection operates correctly with DEBUG logging enabled."""
    from catrans._logging import _get_logger
    catrans_logger = _get_logger()
    old_level = catrans_logger.level
    catrans_logger.setLevel(logging.DEBUG)
    try:
        genes = ["Gene_0", "Gene_1", "Gene_2"]
        samples = [f"Sample_{j}" for j in range(8)]
        counts = pd.DataFrame(
            [
                [5, 6, 5, 4, 50, 48, 52, 50],
                [4, 5, 5, 5, 49, 50, 51, 48],
                [10, 40, 2, 80, 5, 12, 90, 1]
            ],
            index=genes,
            columns=samples
        )
        acute_classification = pd.DataFrame({
            "sample": samples,
            "class": ["Alive"] * 4 + ["Dead"] * 4
        }, index=samples)
        selected = correlation_based_feature_selection(
            data=counts,
            acute_classification=acute_classification,
            classes=CatransClasses(positive="Dead", negative="Alive"),
            cutoff_corr=0.5,
            cutoff_corr_genes=0.5,
            verbose=True
        )
        assert selected is not None
    finally:
        catrans_logger.setLevel(old_level)


def test_run_catrans_feature_selection_failure(monkeypatch: pytest.MonkeyPatch):
    """Verify run_catrans raises ValueError if feature selection returns None."""
    import catrans.core
    monkeypatch.setattr(catrans.core, "correlation_based_feature_selection", lambda **kwargs: None)
    
    counts = pd.DataFrame(np.ones((2, 4)), index=["G0", "G1"], columns=[f"S{i}" for i in range(4)]) 
    meta = pd.DataFrame({"sample": [f"S{i}" for i in range(4)]})
    acute = pd.DataFrame({"sample": [f"S{i}" for i in range(4)], "class": ["Positive", "Negative", np.nan, np.nan]})
    
    with pytest.raises(ValueError, match="Feature selection failed to produce a gene signature"):
        run_catrans(counts, meta, acute, model=LogisticRegression(), signature_genes=None)

