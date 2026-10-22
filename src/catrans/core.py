# Load dependencies
import pandas as pd 
from scipy import stats
import numpy as np
from statsmodels.stats.multitest import multipletests
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
import igraph as ig
from inmoose.pycombat import pycombat_seq
from typing import Any, cast
import logging

from catrans.data import _CatransInput, CatransResult, CatransClasses
from catrans._logging import _get_logger, _setup_internal_logger

logger = _get_logger()

# Functions
# Using instructions in: https://pydeseq2.readthedocs.io/en/stable/auto_examples/plot_step_by_step.html#sphx-glr-auto-examples-plot-step-by-step-py
def _diffExpAnalysis(
    data: pd.DataFrame,
    classification: pd.DataFrame,
    classes: CatransClasses,
    n_cpus: int = 8,
    verbose: bool = False,
    design: str = '~class'
    ) -> tuple[int, pd.DataFrame]:
    """
    Run differential expression analysis using DESeq2-like workflow.

    This function wraps pydeseq2's `DeseqDataSet` and `DeseqStats` to compute
    differential expression between groups in `classification`.

    Parameters:
        data: Gene-expression counts matrix (genes x samples).
        classification: DataFrame with at least columns ['sample', 'class'] describing
            group assignments for samples. Samples with NA class are ignored.
        classes: CatransClasses object defining positive and negative labels.
        n_cpus: Number of parallel workers to use for inference.
        verbose: If True, prints progress messages from pydeseq2 internals.
        design: Formula string for DESeq2 design (e.g. '~class').

    Returns:
        tuple:
            - degs (int): Number of differentially expressed genes (padj < 0.05).
            - dea_res (DataFrame): Full results table from pydeseq2.
    """

    # Filter classification to only include non-NA classes
    non_na_classification = classification.loc[classification['class'].notna(), :].copy()

    # Verify that we have 2 unique classes to run differential expression analysis
    unique_classes = non_na_classification['class'].unique()
    if len(unique_classes) != 2:
        logger.warning(f"Fewer than 2 unique classes present ({list(unique_classes)}). Cannot perform differential expression.")
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
    except Exception:
        raise ValueError('Classification must be a dataframe with a column sample and class')

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
    Calculate pairwise Pearson correlations and p-values for features (columns) in a DataFrame.
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
    Run an iterative semi-supervised model to relabel samples and re-run DE analysis.
    Returns a `CatransResult` object containing iterations and classification history.
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
        except Exception as e:
            logger.error(f"Program has suffered a critical exception: {e}")
            y_pred = None
        

        logger.info("Predictions were successfully performed")
        # Check the prediction
        new_classes = pd.DataFrame(y_pred)
        
        new_classes.set_index(data.columns, inplace=True)
        
        new_classes['prior_classification'] = old_class
        new_classes.columns = ['negative_class_probability','positive_class_probability', 'prior_classification']
        new_classes['current_classification'] = new_classes.apply(
            lambda row: classes.negative if row['negative_class_probability'] >= threshold else (classes.positive if row['positive_class_probability'] >= threshold else np.nan),
            axis = 1
            )
            
        logger.info("Updating classification")

        logger.info(f"The number of unclassified samples in previous iteration is {int(new_classes['prior_classification'].isnull().sum())}")
        logger.info(f"The number of unclassified samples in current iteration is {int(new_classes['current_classification'].isnull().sum())}")
        # Stopping criteria
        if not all(new_classes['prior_classification'] == new_classes['current_classification']) and int(new_classes['prior_classification'].isnull().sum()) > int(new_classes['current_classification'].isnull().sum()):
            
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
    Normalize and batch-correct count data for downstream modeling.

    This function applies ComBat-seq batch correction on raw counts and then
    uses pydeseq2 to compute normalized counts, optionally applying log1p.

    Parameters:
        counts_df: Raw count matrix (genes x samples).
        metadata_df: Sample metadata containing at least 'sample' and batch column.
        batch_col: Column name in metadata specifying batch assignments.
        log_transform: If True, apply log1p to normalized counts.

    Returns:
        DataFrame of normalized (and optionally log-transformed) counts.
    """
    logger.info("Performing Data Normalization")

    if batch_col:     
        logger.info(f"Batch column input: {batch_col}")        
        logger.info("Performing batch correction")
        # pycombat expects (genes x samples)
        batch = metadata_df[batch_col].tolist()
        data_for_dds = pycombat_seq(counts_df, batch).T
    else:
        logger.info("No batch column input")
        logger.info("Normalization will be performed without batch correction")
        
        data_for_dds = counts_df.T

    inference = DefaultInference(n_cpus=1) 
    dds = DeseqDataSet(
        counts=pd.DataFrame(data_for_dds), # pd.df just to shut up pylance error
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
        classes: CatransClasses = CatransClasses(),
        pval: float = 0.05,
        cutoff_corr: float | None = None,
        cutoff_corr_genes: float | None = None,
        verbose: bool = False
    ) -> list[str] | None:
    """
    Find gene-gene correlation and select representative features.

    This function computes gene-wise Spearman correlation against the class label,
    then builds an undirected graph of highly correlated genes and selects a
    representative gene per connected component.

    Parameters:
        data: Gene-expression matrix (genes x samples).
        acute_classification: DataFrame with columns ['sample', 'class'] representing
            the class label for each sample.
        classes: CatransClasses object defining positive and negative labels.
        pval: P-value threshold for significance when filtering gene-gene correlations.
        cutoff_corr: Minimum absolute Spearman R for including genes in the gene-gene graph.
        cutoff_corr_genes: Minimum absolute Pearson R for edge inclusion in the graph.

    Returns:
        DataFrame with selected genes and their correlation statistics.
    """

    _setup_internal_logger(verbose)

    logger.info("Initializing Correlation-Based Feature Learning")

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
        logger.warning("By default the analyses will be performed only with significant correlations (p-value <= 0.05)") 

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
    
    