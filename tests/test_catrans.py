import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from catrans.data import CatransResult, CatransClasses
from catrans.core import (
    _get_normalized_dataset,
    correlation_based_feature_selection,
    run_catrans,
)

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
    
    # Set final classification
    final_df = pd.DataFrame({"class": ["Alive", "Dead"]})
    result._set_final_classification(final_df)
    assert result.final_classification is not None
    assert result.final_classification.iloc[0]["class"] == "Alive"

def test_normalization_without_batch_correction():
    """Verify that _get_normalized_dataset performs median-of-ratios scale-normalization."""
    # 5 genes, 4 samples
    counts = pd.DataFrame(
        [[10, 20, 15, 30],
         [5, 10, 8, 12],
         [100, 200, 150, 280],
         [2, 4, 3, 5],
         [0, 1, 0, 2]],
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
    selected = correlation_based_feature_selection(
        data=counts,
        acute_classification=acute_classification,
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
    
    result = run_catrans(
        raw_counts=raw_counts,
        metadata=metadata,
        acute_classification=acute_classification,
        model=model,
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
