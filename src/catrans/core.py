# Load dependencies
import logging
from typing import Any, cast

import igraph as ig
import numpy as np
import pandas as pd
from inmoose.pycombat import pycombat_seq
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from scipy import stats
from statsmodels.stats.multitest import multipletests

from catrans._logging import _get_logger, _setup_internal_logger
from catrans.data import CatransClasses, CatransResult, _CatransInput

logger = _get_logger()

# Functions
# Using instructions in: https://pydeseq2.readthedocs.io/en/stable/auto_examples/plot_step_by_step.html#sphx-glr-auto-examples-plot-step-by-step-py
def _diffExpAnalysis(
    data: pd.DataFrame,
    classification: pd.DataFrame,
    classes: CatransClasses,
    n_cpus: int = 8,
    design: str = '~class',
    verbose: bool = False
    ) -> tuple[int, pd.DataFrame]:
    """
    Run differential expression analysis using a DESeq2 workflow via PyDESeq2.

    This function wraps PyDESeq2's `DeseqDataSet` and `DeseqStats` to compute negative
    binomial dispersion estimates and perform Wald testing between the positive and
    negative classes. Samples with NA or missing class labels are automatically filtered
    out prior to model fitting.

    Parameters:
        data: Gene-expression raw counts matrix of shape (genes x samples).
        classification: DataFrame with at least columns ['sample', 'class'] describing
            group assignments for samples. Samples with NA class are ignored.
        classes: CatransClasses instance defining positive and negative condition labels.
        n_cpus: Number of parallel CPU workers to use for PyDESeq2 dispersion fitting.
        design: Design formula string for DESeq2 (default: '~class').
        verbose: If True, prints progress and diagnostic messages from PyDESeq2 internals.

    Returns:
        tuple[int, pd.DataFrame]:
            - degs (int): Number of statistically significant differentially expressed
              genes with adjusted p-value (padj) < 0.05.
            - dea_res (pd.DataFrame): Complete differential expression statistics table
              from PyDESeq2 containing baseMean, log2FoldChange, lfcSE, stat, pvalue, and padj.
    """

    # Filter classification to only include non-NA classes
    non_na_classification = classification.loc[classification['class'].notna(), :].copy()

    # Verify that we have 2 unique classes to run differential expression analysis
    unique_classes = non_na_classification['class'].unique()
    if len(unique_classes) != 2:
        logger.warning(
            f"Fewer than 2 unique classes present ({list(unique_classes)}). "
            "Cannot perform differential expression."
        )
        return 0, pd.DataFrame()

    # Convert the class column to categorical with levels [negative, positive] to fix Wald contrast direction
    non_na_classification['class'] = pd.Categorical(
        non_na_classification['class'],
        categories=[classes.negative, classes.positive]
    )

    logger.info("Loading data...")
    logger.info(f"The dimensions of data are: {data.shape}.")
        
    try:
        dat_c = data.loc[:, non_na_classification["sample"]]
    except KeyError as err:
        raise ValueError('Classification must be a dataframe with a column sample and class') from err

    logger.info(f"The dimensions of classified data are: {dat_c.shape}.")
    
    logger.info("Running Differential Expression Analysis")

    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=dat_c.T,
        metadata=non_na_classification,
        design = design,
        refit_cooks = True,
        inference = inference,
        quiet = not verbose
    )

    dds.fit_size_factors()
    dds.fit_genewise_dispersions()
    dds.fit_dispersion_trend()
    dds.fit_dispersion_prior()
    dds.fit_MAP_dispersions()
    dds.fit_LFC()

    # Potential outliers
    dds.calculate_cooks()
    if dds.refit_cooks:
        dds.refit()
    
    ds = DeseqStats(
        dds,
        contrast = np.array([0,1]),
        cooks_filter=True,
        independent_filter = True,
        quiet = not verbose
    )

    ds.run_wald_test()
    
    # Adjusted p-valor
    if ds.independent_filter:
        ds._independent_filtering()
    else:
        ds._p_value_adjustment()

    
    logger.info("Differential Expression Analysis was successfully performed.")

    # Results
    ds.summary()
    dea_res = ds.results_df
    logger.debug(f"Differential Expression Analysis results:\n{dea_res.head(20).to_string()}")
    degs = dea_res.query('padj < 0.05').shape[0]

    return degs, dea_res


def _correlation_of_gene_tuples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate pairwise Pearson correlations and parametric p-values for candidate gene pairs.

    Computes all pairwise Pearson correlation coefficients (r) between candidate genes across
    samples, and calculates two-tailed p-values using Student's t-distribution with degrees
    of freedom df = n - 2. Self-correlations and redundant reciprocal pairs (e.g. Gene A - Gene B
    vs. Gene B - Gene A) are excluded by selecting only the upper triangle of the correlation matrix.

    Parameters:
        df: Expression matrix of shape (n_samples, n_candidate_genes), where columns represent
            candidate gene identifiers and rows represent individual biological samples.

    Returns:
        pd.DataFrame: Table of unique pairwise gene correlations with columns:
            - 'Gene1' (str): Identifier of the first gene in the pair.
            - 'Gene2' (str): Identifier of the second gene in the pair.
            - 'R' (float): Pearson correlation coefficient between Gene1 and Gene2.
            - 'p_value' (float): Two-tailed parametric p-value testing the null hypothesis
               that the true correlation is zero.
    """

    # Calculate Pearson correlation (r)
    r_df = df.corr(method='pearson')
    # Calculate p-values with numpy 
    n = len(df)
    df_degrees = n - 2
    with np.errstate(divide='ignore', invalid='ignore'):
        t_matrix = r_df * np.sqrt(df_degrees / (1 - r_df**2))
        p_matrix = stats.t.sf(np.abs(t_matrix), df_degrees) * 2

    np.fill_diagonal(p_matrix, np.nan)
    p_df = pd.DataFrame(p_matrix, index=r_df.index, columns=r_df.columns)

    # Avoid repeted combinations by only having from the top diagonal
    mask = np.triu(np.ones(r_df.shape), k=1).astype(bool)
    df_result = pd.DataFrame({
        'Gene1' : np.repeat(r_df.index, r_df.shape[1])[mask.ravel()],
        'Gene2' : np.tile(r_df.index, r_df.shape[0])[mask.ravel()],
        'R' : r_df.values[mask],
        'p_value' : p_df.values[mask]
    })
    return df_result


def _semisupervised_model(
        data_raw: pd.DataFrame,
        data: pd.DataFrame,
        signature: list[str],
        acute_classification: pd.DataFrame,
        model: Any,
        threshold: float,
        classes: CatransClasses,
        design: str = 'class',
        n_cpus: int = 8
        ) -> CatransResult:
    """
    Execute the iterative semi-supervised self-training loop to refine sample stratification.

    This function trains a probabilistic supervised classifier on currently labeled samples
    using normalized expression values of the selected biomarker signature. It predicts class
    probabilities for all samples (including unlabeled / intermediate cases), relabels samples
    whose posterior class probability meets or exceeds `threshold`, and performs Differential
    Expression Analysis (DEA) with PyDESeq2 using the raw count matrix on the newly classified cohorts.

    Parameters:
        data_raw: Unnormalized integer count matrix of shape (genes x samples) used strictly
            for negative binomial dispersion fitting and differential expression analysis in PyDESeq2.
        data: Normalized (and optionally batch-corrected) expression matrix of shape (genes x samples)
            used as feature input for model training and prediction.
        signature: List of gene identifiers forming the non-redundant predictive signature.
        acute_classification: DataFrame indexed by sample identifier containing initial class
            assignments in the column specified by `design`. Intermediate/unclassified samples
            should contain NaN.
        model: Supervised classifier instance implementing scikit-learn compatible `fit(X, y)`
            and `predict_proba(X)` methods.
        threshold: Minimum posterior probability cutoff (e.g. 0.8) required to assign an
            unclassified sample to either the positive or negative class.
        classes: CatransClasses instance defining positive and negative condition labels.
        design: Column name in `acute_classification` representing condition labels (default: 'class').
        n_cpus: Number of parallel CPU workers to use for PyDESeq2 computations.

    Returns:
        CatransResult: Results container storing the sequential history of all successful
            iterations (iteration index, DEG counts, full PyDESeq2 tables, prediction probabilities)
            and the final sample classification.
    """

    results = CatransResult()
    signal = True
    old_class = acute_classification[design]
    i = 0
    # baseline for degs; this way we can infer if classifications are indeed "better" than the initial classification.
    degs, _ = _diffExpAnalysis(
        data = data_raw, 
        classification = acute_classification,
        classes = classes,
        n_cpus = n_cpus,
        design= "~"+design
    )

    while signal:

        i += 1
        logger.info(f"Iteration {i}")

        # Create the data frame with normalised and batch corrected data and the class for each sample
        # Filter to only labeled samples to avoid training on unlabeled (NaN) samples
        labeled_samples = old_class.dropna().index
        X_tr = data.loc[signature, labeled_samples].T.to_numpy()
        y_tr = old_class.loc[labeled_samples]
        y_tr = y_tr.map(lambda x: 1 if x == classes.positive else 0).to_numpy()

        # Train the model and predict
        logger.info("Generating new predictions")

        try: 
            model.fit(X_tr, y_tr)
            y_pred = model.predict_proba(data.loc[signature, :].T.to_numpy())
        except Exception as e:  # noqa: BLE001
            logger.error(f"Program has suffered a critical exception: {e}")
            y_pred = None

        logger.info("Predictions were successfully performed")
        # Check the prediction
        new_classes = pd.DataFrame(y_pred)
        new_classes.set_index(data.columns, inplace=True)
        new_classes['prior_classification'] = old_class
        new_classes.columns = ['negative_class_probability', 'positive_class_probability', 'prior_classification']
        new_classes['current_classification'] = new_classes.apply(
            lambda row: (
                classes.negative
                if row['negative_class_probability'] >= threshold
                else (classes.positive if row['positive_class_probability'] >= threshold else np.nan)
            ),
            axis=1,
        )

        logger.info("Updating classification")

        prior_unclass = int(new_classes['prior_classification'].isnull().sum())
        curr_unclass = int(new_classes['current_classification'].isnull().sum())

        logger.info(f"The number of unclassified samples in previous iteration is {prior_unclass}")
        logger.info(f"The number of unclassified samples in current iteration is {curr_unclass}")

        # Stopping criteria
        class_changed = not all(new_classes['prior_classification'] == new_classes['current_classification'])
        if class_changed and prior_unclass > curr_unclass:
            
            # Update results object incrementally
            logger.info("Classification changed; running Differential Expression Analysis...")        
            classification = pd.DataFrame({
                "sample" : new_classes.loc[new_classes['current_classification'].notna(),:].index,
                design : new_classes.loc[new_classes['current_classification'].notna(),"current_classification"]
            })
            logger.info("Performing Differential Expression Analysis with new classification.")

            new_degs, dea_results = _diffExpAnalysis(
                data = data_raw, 
                classification = classification,
                classes = classes,
                n_cpus = n_cpus,
                design= "~"+design
            )
            
            logger.info(f"DEGs on prior iteration: {degs}")
            logger.info(f"DEGs on current iteration: {new_degs}")
            
            if new_degs > degs:
                
                logger.info("All conditions passed, saving iteration result.")
                results._add_iteration(i, new_degs, dea_results, new_classes)
                old_class = new_classes['current_classification']
                degs = new_degs
            
            else:
                logger.info("Stopping condition met: no more DEGs discovered than in previous step.")
                logger.info("Results of this iteration will not be saved.")
                
                signal = False
            
        else:
            logger.info("Stopping condition met: No changes in classification or no new samples classified.")
            logger.info("Results of this iteration will not be saved.")
            
            signal = False

    logger.info("Program ended.")
    final_df = pd.DataFrame(old_class)
    final_df.columns = [design]
    results._set_final_classification(final_df)


    return results


def _get_normalized_dataset(
    counts_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    batch_col: str | None = None,
    log_transform: bool = True,
    ) -> pd.DataFrame:
    """
    Normalize raw RNA-seq counts and optionally adjust for technical batch effects.

    When `batch_col` is provided, raw counts are adjusted for known technical batch effects
    using ComBat-seq (`pycombat_seq`). Median-of-ratios size factor estimation is then performed
    via PyDESeq2's `DeseqDataSet`, scaling counts across samples. Finally, an optional natural
    logarithmic transformation ln(1 + x) (`np.log1p`) is applied to stabilize variance for
    downstream machine learning models.

    Parameters:
        counts_df: Raw integer count matrix of shape (genes x samples).
        metadata_df: Sample metadata DataFrame containing sample identifiers and batch annotations.
        batch_col: Column name in `metadata_df` identifying the batch/run variable. If None,
            batch correction is skipped and counts are normalized directly.
        log_transform: If True (default), applies ln(1 + x) transformation to normalized counts.

    Returns:
        pd.DataFrame: Normalized (and optionally log-transformed) expression matrix of shape
            (genes x samples), preserving original gene and sample labels.
    """
    logger.info("Performing Data Normalization")

    if batch_col:     
        logger.info(f"Batch column input: {batch_col}")        
        logger.info("Performing batch correction")
        # pycombat expects (genes x samples)
        batch = metadata_df[batch_col].to_list()
        data_for_dds = pycombat_seq(counts_df, batch).T
    else:
        logger.info("No batch column input")
        logger.info("Normalization will be performed without batch correction")
        
        data_for_dds = counts_df.T

    inference = DefaultInference(n_cpus=1) 
    dds = DeseqDataSet(
        counts=pd.DataFrame(data_for_dds),
        metadata=metadata_df,
        design='~1',
        inference=inference,
        quiet=True
    )   

    dds.fit_size_factors()

    normed_counts = pd.DataFrame(
        dds.layers["normed_counts"],
        index=pd.DataFrame(data_for_dds).index,
        columns=pd.DataFrame(data_for_dds).columns,
    )

    if log_transform:
        normed_counts = pd.DataFrame(np.log1p(normed_counts))

    logger.info("Normalized data successfully")

    return normed_counts.T


def correlation_based_feature_selection(
        data: pd.DataFrame,
        acute_classification: pd.DataFrame,
        classes: CatransClasses | None = None,
        pval: float = 0.05,
        cutoff_corr: float | None = None,
        cutoff_corr_genes: float | None = None,
        verbose: bool = False
    ) -> list[str] | None:
    """
    Perform correlation-based feature selection to extract a non-redundant biomarker signature.

    Identifies candidate genes significantly correlated with the acute clinical phenotype,
    then builds an undirected graph where nodes represent candidate genes and edges denote high
    pairwise correlation (Pearson |r| >= `cutoff_corr_genes`). For each connected component in
    the graph, the gene exhibiting the highest correlation with the phenotype is selected as the
    component's representative, effectively removing collinearity and redundancy.

    Parameters:
        data: Gene-expression matrix of shape (genes x samples).
        acute_classification: DataFrame with columns ['sample', 'class'] representing the binary
            phenotype assignments. Intermediate or unclassified samples (NaN) are ignored.
        classes: CatransClasses instance defining positive and negative condition labels.
            If None, defaults to Positive="Positive" and Negative="Negative".
        pval: Significance threshold for FDR-adjusted (Benjamini-Hochberg) p-values when
            filtering candidate phenotype-correlated genes (default: 0.05).
        cutoff_corr: Minimum absolute Spearman correlation coefficient (|r|) with the class label
            required for candidate gene selection. If None, defaults to the 90th percentile of
            absolute correlations across all genes.
        cutoff_corr_genes: Minimum absolute Pearson correlation coefficient (|r|) between candidate
            genes to establish an edge in the de-redundancy graph. If None, defaults to the 95th
            percentile of pairwise gene correlations.
        verbose: If True, enables detailed log messages during the feature selection process.

    Returns:
        list[str] | None: List of selected non-redundant gene identifiers, or None if no candidate
            genes meet the correlation and significance thresholds.
    """

    _setup_internal_logger(verbose)

    logger.info("Initializing Correlation-Based Feature Learning")

    # check classes in case of defaulting
    if classes is None:
        classes = CatransClasses()

    classified_mask = acute_classification['class'].notna()
    classified_samples_acute = acute_classification.loc[classified_mask, 'sample']

    # Samples are columns
    dat_c = data.loc[:, classified_samples_acute]

    corr_list = []
    # Map labels to numeric to ensure scipy spearmanr calculates correctly on consistent numeric values
    y_class = acute_classification.loc[classified_mask, 'class'].map({classes.negative: 0, classes.positive: 1})
    logger.info("Performing correlation between Acute classification and genes expression")

    for i in range(dat_c.shape[0]):
        res = stats.spearmanr(dat_c.iloc[i, :], y_class)
        corr_list.append({
            'rho': res[0],
            'p_value': res[1]
        })

    corr1 = pd.DataFrame(corr_list, index=dat_c.index)
    corr1['Genes'] = corr1.index

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"corr1:\n{corr1.head(n=200).to_string()}")

    if not cutoff_corr:
        logger.info("Correlation between genes expression and the initial classification is not defined")
        logger.info("Using Q3 of absolute rho values as cutoff")
        cutoff_corr = abs(corr1['rho']).quantile(0.75)
    
    logger.info(f"R cutoff value between genes expression and classification: {cutoff_corr:.3f}")  
    corr_genes = corr1[(abs(corr1['rho'])>=cutoff_corr) & (corr1['p_value']<=pval)].index.to_list()
    logger.info(f"Number of significant genes with R higher than {cutoff_corr:.3f}: {len(corr_genes)}")

    if len(corr_genes) == 0:

        logger.warning(f"There are no genes higher than {cutoff_corr}.")
        logger.warning(f"The maximum R value is {np.round(np.max(abs(corr1['rho'])),2)}.") # Fixed typo: 'R' to 'rho'
        logger.warning(
            "By default the analyses will be performed only with significant correlations (p-value <= 0.05)"
        )

        corr_genes = corr1[corr1['p_value']<=pval].index.to_list()

    if len(corr_genes) <= 1:
        logger.info(f"Fewer than 2 genes selected ({len(corr_genes)}); skipping gene-gene network reduction.")
        return cast(list[str], corr_genes)

    subdat = dat_c.loc[corr_genes,:]

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Selecction of genes data:\n{subdat.head(n=200).to_string()}")
    logger.debug(f"Shape: {subdat.shape}")
    logger.info("Performing Gene to Gene correlation for Feature Selection")

    corr2 = _correlation_of_gene_tuples(subdat.T)
    # pval_correction_method = 'fdr_bh'
    pval_correction_method = 'bonferroni'
    logger.debug(f"Using {pval_correction_method} to correct p-values")
    _, corr2['p_adj'], _, _ = multipletests(corr2['p_value'], alpha=0.05, method=pval_correction_method)
    

    # if logger.level == logging.DEBUG:
        # corr1.to_csv("corr1.csv", index=False, header = True) # DEBUG
        # corr2.to_csv("corr2.csv", index=False, header = True) # DEBUG
        # corr_2_2.to_csv("corr2_2.csv", index=False, header = True) # DEBUG
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"correlations Gene to Gene Results:\n{corr2.head(n=200).to_string()}")
    
    # Using iGraph for gene selection. 
    # Correlation threshold is used to build a graph. 
    # We only select the most representative (higher R with CLASSIFICATION) gene from each set

    if not cutoff_corr_genes:
        percentile = 0.75
        logger.info("Correlation between genes expression is not defined")
        logger.info("Using Q3 of absolute R values as cutoff")
        # logger.info(f"Using percentile {percentile} of absolute R values as cutoff")
        cutoff_corr_genes = abs(corr2['R']).quantile(percentile)
        
    logger.info(f"R Gene x Gene cutoff value: {cutoff_corr_genes:.3f}") # NOTE: change wording

    gene_tuples = (
        corr2[
            (corr2['p_adj'] < pval)
            & (abs(corr2['R']) > cutoff_corr_genes)
        ]
        .loc[:, ['Gene1', "Gene2"]]
        .itertuples(index=False)  # Generator
    )
    
    # Graph building
    g = ig.Graph.TupleList(gene_tuples)
    components = g.connected_components(mode='weak')
    all_genes = set(corr2['Gene1']).union(set(corr2['Gene2']))
    current_nodes = set(g.vs['name'])
    isolated_nodes = all_genes - current_nodes
    g.add_vertices(list(isolated_nodes))
    components = g.connected_components(mode='weak')
    
    toselection = corr1.loc[g.get_vertex_dataframe()['name'],:]
    toselection['membership'] = components.membership
    selected_genes = []

    for i in set(toselection['membership']):
        logger.debug(f"Cluster: {i}")
        values = toselection.loc[toselection['membership'] == i]
        logger.debug(f"Number of genes in Cluster {i}: {len(values)}")
        logger.debug(f"Cluster values: \n{values.to_string()}")
        selected_gene = abs(values['rho']).idxmax()
        logger.debug(f"Gene with max absolute R value from Cluster {i}: {selected_gene}")
        selected_genes.append(selected_gene)
       
    corr1_final = corr1.loc[(corr1['Genes'].isin(selected_genes) & (corr1['p_value'] <= pval))] # is pval needed? 
    logger.debug(corr1_final.to_string())
    logger.info(f"Number of final genes selected: {len(corr1_final)}")
    
    return cast(list[str], corr1_final['Genes'].to_list())

    
def run_catrans(
        raw_counts: pd.DataFrame,
        metadata: pd.DataFrame,
        acute_classification: pd.DataFrame,
        model: Any,
        classes: CatransClasses | None = None,
        batch_col: str | None = None,
        model_threshold: float = 0.8,
        n_cpus: int = 8,
        signature_genes: list[str] | None = None,
        verbose: bool = False
    ) -> CatransResult:
    """
    Run the complete CATranS (Clinical Acute Transcriptomic Stratification) pipeline.

    CATranS implements an iterative semi-supervised learning framework to refine clinical sample
    stratification. Starting from a core subset of extreme "acute" cases (e.g. extreme survivors
    vs. non-survivors), it iteratively predicts class probabilities for intermediate or unclassified
    samples using a supervised machine learning model. High-confidence predictions are pseudo-labeled,
    and each iteration is biologically validated by performing differential expression analysis
    (PyDESeq2) on raw counts to ensure the transcriptional contrast between cohorts strictly increases.

    Workflow:
        1. Validate input types, DataFrame schemas ('sample', 'class'), and model capabilities.
        2. If `signature_genes` is not provided, perform correlation-based feature selection
           (`correlation_based_feature_selection`) to identify a non-redundant biomarker signature.
        3. Normalize expression counts using PyDESeq2 median-of-ratios scaling and optionally
           correct technical batch effects using ComBat-seq (`_get_normalized_dataset`).
        4. Execute iterative self-training (`_semisupervised_model`) until convergence or until
           differential expression power ceases to improve.

    Parameters:
        raw_counts: Raw integer read count matrix of shape (genes x samples).
        metadata: Sample metadata DataFrame containing sample identifiers and optional batch covariates.
        acute_classification: DataFrame with required columns ['sample', 'class'] specifying initial
            acute phenotype assignments. Intermediate or unclassified samples must contain NaN in 'class'.
        model: Supervised classification model instance implementing scikit-learn compatible `fit(X, y)`
            and `predict_proba(X)` methods (e.g. `RandomForestClassifier`, `LogisticRegression`).
        classes: CatransClasses instance defining the positive and negative label strings.
            If None, defaults to Positive="Positive" and Negative="Negative".
        batch_col: Optional column name in `metadata` specifying technical batches for ComBat-seq correction.
        model_threshold: Posterior probability cutoff (0.5 < threshold <= 1.0) required to relabel an
            unclassified sample during self-training iterations (default: 0.8).
        n_cpus: Number of parallel CPU workers to use for PyDESeq2 dispersion fitting (default: 8).
        signature_genes: Optional pre-selected list of gene identifiers to use as predictive features.
            If None, feature selection is performed automatically using default parameters.
        verbose: If True, enables detailed progress and diagnostic logging to standard output.

    Returns:
        CatransResult: Results object containing:
            - `iterations`: List of `_IterationResult` objects tracking DEA tables, DEG counts,
              and model prediction probabilities for every successful iteration.
            - `final_classification`: DataFrame containing the final converged sample assignments.
            - `number_of_iterations`: Total count of successful relabeling iterations performed.
            - `last_iteration`: Shortcut to the final iteration's results.

    Raises:
        TypeError: If `acute_classification` is not a pandas DataFrame.
        ValueError: If `acute_classification` lacks 'sample' or 'class' columns, or contains
            labels other than `classes.positive`, `classes.negative`, or NaN.
        AttributeError: If `model` lacks a `predict_proba` method.
    """

    _setup_internal_logger(verbose=verbose)

    if classes is None:
        classes = CatransClasses()

    # Argument checking
    if not isinstance(acute_classification, pd.DataFrame):
        raise TypeError("acute_classification must be a pandas.DataFrame")

    if 'sample' not in acute_classification.columns or 'class' not in acute_classification.columns:
        raise ValueError("acute_classification must be a pandas.DataFrame with 'sample' and 'class' columns.")

    # Validate classes in acute_classification
    allowed_values = {classes.positive, classes.negative}
    actual_values = set(acute_classification['class'].dropna().unique())
    invalid_values = actual_values - allowed_values
    if invalid_values:
        raise ValueError(
            f"Found invalid labels in acute_classification['class']: {invalid_values}. "
            f"Only '{classes.positive}' (positive) and '{classes.negative}' (negative) are allowed, plus NaN."
        )

    # Create a copy to avoid mutating the caller's DataFrame in-place
    acute_classification = acute_classification.copy()

    acute_classification.set_index(
        pd.Index(acute_classification['sample'].to_list()),
        inplace=True
        )

    if not hasattr(model, 'predict_proba'):
        raise AttributeError("Passed model must have a predict_proba method in order to run semi-supervised learning.")

    if signature_genes is None:
        logger.info("No signature genes were provided.")
        logger.info("Feature selection will be performed with default parameters.")
        
        signature_genes = correlation_based_feature_selection(
            data=raw_counts,
            acute_classification=acute_classification,
            classes=classes,
            verbose=verbose
        )
        
        if signature_genes is None:
            raise ValueError("Feature selection failed to produce a gene signature.")

    input_data = _CatransInput(
        raw_counts=raw_counts,
        metadata=metadata,
        acute_classification=acute_classification,
        model=model,
        classes=classes,
        batch_col=batch_col,
        model_threshold=model_threshold,
        n_cpus=n_cpus,
        signature_genes=signature_genes
    )

    logger.info("Initializing CATranS algorithm")
    logger.info("Normalizing DataSet...")

    normalized_data = _get_normalized_dataset(
        counts_df=input_data.raw_counts,
        metadata_df=input_data.metadata,
        batch_col=input_data.batch_col
    )

    results = _semisupervised_model(
        data_raw=input_data.raw_counts,
        data=normalized_data, 
        signature=input_data.signature_genes or [],
        acute_classification=input_data.acute_classification,
        model=input_data.model,
        threshold=input_data.model_threshold,
        classes=input_data.classes,
    )
    
    return results
    
    