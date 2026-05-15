# Load dependencies
import pandas as pd 
from scipy import stats
import statistics as st
import numpy as np
import scipy.stats as sc
from statsmodels.stats.multitest import multipletests
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
import igraph as ig
from itertools import combinations
from multiprocessing import Pool
from inmoose.pycombat import pycombat_seq
from typing import Any, List
import logging
from time import time

# internal dependencies
from catrans._logging import _get_logger, _setup_internal_logger

logger = _get_logger()

# Functions
# Using instructions in: https://pydeseq2.readthedocs.io/en/stable/auto_examples/plot_step_by_step.html#sphx-glr-auto-examples-plot-step-by-step-py
def _diffExpAnalysis(
    data: pd.DataFrame,
    classification: pd.DataFrame,
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
        n_cpus: Number of parallel workers to use for inference.
        verbose: If True, prints progress messages from pydeseq2 internals.
        design: Formula string for DESeq2 design (e.g. '~class').

    Returns:
        tuple:
            - degs (int): Number of differentially expressed genes (padj < 0.05).
            - dea_res (DataFrame): Full results table from pydeseq2.
    """

    # print(f'The dimensions of data are: {data.shape}\n')
    # print(f'The dimensions of experimental design are: {ed.shape}\n')
    # _setup_internal_logger(verbose=verbose)

    logger.info(f"Loading data...")
    logger.info(f"The dimensions of data are: {data.shape}.")
        
    try:
        dat_c = data.loc[:, classification.loc[classification['class'].notna(),"sample"]]
    except Exception:
        raise ValueError('Classification must be a dataframe with a column sample and class') # NOTE: change if data structure changes. Or even before, this wasn't working isn't it?


    logger.info("Running Differential Expression Analysis")

    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=dat_c.T,
        metadata=classification,
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
    logging.info(f"Differential Expression Analysis results:\n{ds.summary()}") # NOTE: It is possible that I will change this to debug level

    ds.summary() # NOTE: where does this goes?
    dea_res = ds.results_df
    degs = dea_res.query('padj < 0.05').shape[0]

    return degs, dea_res


# def _corr_loop(mytupleindex):
#     """Helper for parallel correlation computation between two genes.

#     This function is intended to be used with multiprocessing.Pool.map.

#     Args:
#         mytupleindex: Tuple of (subdat, i, j) where `subdat` is a gene x sample
#             DataFrame and i/j are row indices to correlate.

#     Returns:
#         DataFrame with columns ['Gene1', 'Gene2', 'R', 'p_value'] for the pair.
#     """
#     subdat, i, j = mytupleindex
#     res = sc.pearsonr(
#         subdat.iloc[i,:],
#         subdat.iloc[j,:]
#         )
#     res_row = pd.DataFrame.from_dict({
#         'Gene1': [subdat.index[i]],
#         'Gene2':[subdat.index[j]],  
#         'R': [res.statistic],
#         'p_value': [res.pvalue]})
#     return res_row

def _correlation_of_gene_tuples(df: pd.DataFrame):
    """
    doc string
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
        data: pd.DataFrame,
        data_raw: pd.DataFrame,
        signature: list[str],
        acute_classification: pd.DataFrame,
        models: list[Any],
        threshold: float,
        design: str = 'class') -> tuple[pd.DataFrame, list[pd.DataFrame], list[int]]:
    """
    Run an iterative semi-supervised model to relabel samples and re-run DE analysis.

    The function trains classifiers on labeled data, predicts classes for all samples,
    updates labels for confidently predicted samples, re-runs differential expression
    analysis, and repeats until convergence.

    Parameters:
        data: Normalized (and batch-corrected) expression data (genes x samples).
        data_raw: Raw count matrix (genes x samples) used for differential expression.
        signature: List of genes/features used for model training.
        acute_classification: DataFrame with columns ['sample', 'class'] for initial labels.
        models: List of sklearn-like classifiers supporting fit() and predict_proba().
        threshold: Probability cutoff used to label a sample as Alive or Dead.
        design: Design formula for DESeq2 when re-running differential expression.

    Returns:
        tuple:
            - history (DataFrame): Classification probability history for each iteration.
            - new_results_df (DataFrame): Last differential expression results. 
            - degs_list (list[int]): Number of DE genes found at each iteration.
    """
    
    logger.info("Starting semi-supervised classification...")

    signal = True
    old_class = acute_classification[design]
    degs = 0
    i = 0
    history = pd.DataFrame()
    degs_list = []
    dea_results_list = []


    while signal:

        i += 1
        logger.info(f"Iteration {i}")

        # Create the data frame with normalised and batch corrected data and the class for each sample
        X_tr = data.loc[signature, old_class.notna().index].T.to_numpy()
        y_tr = old_class
        y_tr = y_tr.map(lambda x: 1 if x == "Dead" else 0).to_numpy()

        # Train the model and predict
        logger.info(f"Generating new predictions")

        ## Here it should be some try-except logic
        try:
            if len(models) > 1:
                
                predictions = []
                for model in models:
                    model.fit(X_tr, y_tr)
                    y_pred_sub = model.predict_proba(data.loc[signature,:].T.to_numpy())
                    predictions.append(y_pred_sub)

                y_pred = np.mean(predictions, axis = 0)

            else:
                models[0].fit(X_tr, y_tr)
                y_pred = models[0].predict_proba(data.loc[signature, :].T.to_numpy())

        # TODO: study what exceptions we could have here instead of a generic Exception
        except Exception as e:
            logger.critical(f"Program has suffered a critical exception: {e}")

        logger.info(f"Predictions were successfully performed")
        # Check the prediction
        new_classes = pd.DataFrame(y_pred)
        
        new_classes.set_index(data.columns, inplace=True)
        
        new_classes['old_class'] = old_class
        new_classes.columns = ['ap','dp', 'old_class']
        new_classes['new_class'] = new_classes.apply(lambda row: 'Alive' if row['ap'] >= threshold else ('Dead' if row['dp'] > threshold else np.nan), axis = 1) # por culpa de esto hay NAs en valores ya etiquetados, pero es algo que puede ocurrir

        # Save predictions
        classes_history = new_classes.copy()
        classes_history.columns = [f"ap_{i}",f"dp_{i}",f"old_class_{i}", f"new_class_{i}"]
        history = pd.concat([history,classes_history], axis=1)


        # Stopping criteria
        if not all(new_classes['old_class'] == new_classes['new_class']) and int(new_classes['old_class'].isnull().values.sum()) > int(new_classes['new_class'].isnull().values.sum()):
            logger.info("No changes in classification in this iteration")
            logger.info("Stopping")
            
            classification = pd.DataFrame({"sample" : new_classes.loc[new_classes['new_class'].notna(),:].index,
                                            design : new_classes.loc[new_classes['new_class'].notna(),"new_class"]}) # esta columna se debería llamar design, ya que entra en la siguiente función, si alguien no le apetece llamarle class no funcionaría #TODO #DONE
            
            new_degs, results_df = _diffExpAnalysis(
            data = data_raw, 
            classification = classification,
            n_cpus = 8, # May be changed to a parameter if we want to make it more flexible # TODO
            design= "~"+design
            )

                            
            degs_list.append(new_degs)
            dea_results_list.append(results_df)
            
            if new_degs > degs:
                
                old_class = new_classes['new_class']
                degs = new_degs
            
            else:
                print("condición de parada 1 y 2")
                signal = False
            
        else:
            logger.info("No new Differentially Expressed Genes discovered")
            logger.info("Stopping")
            signal = False

        # NOTE: should there be a positive condition to inform the algorithm will keep working? Or an ealry return on the conditional tree ?

        
        # NOTE: modifiqué la identación para que salga siempre, pero ya si eso modificaré lo de que solamente pille la última iteración
        # NOTE: I modified the identation in order to see these messages always. However, it is possible I will modify this so it only appears on the last iteration with the final result

        logger.info(f"The number of unclassified samples in old class is {int(new_classes['old_class'].isnull().values.sum())}")
        logger.info(f"The number of unclassified samples in new class is {int(new_classes['new_class'].isnull().values.sum())}")
        
    return history, degs_list, dea_results_list


def _get_normalized_batch_corrected(
    counts_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    # design_factor: str = "1", # será class, no? # NOTE: it isn't necessary since design_factor is 1
    batch_col: str | None = None,
    log_transform: bool = True,
    ) -> pd.DataFrame | Any:
    """
    Normalize and batch-correct count data for downstream modeling.

    This function applies ComBat-seq batch correction on raw counts and then
    uses pydeseq2 to compute normalized counts, optionally applying log1p.

    Parameters:
        counts_df: Raw count matrix (genes x samples).
        metadata_df: Sample metadata containing at least 'sample' and batch column.
        design_factor: Column name in metadata to use as the design factor for normalization.
        batch_col: Column name in metadata specifying batch assignments.
        log_transform: If True, apply log1p to normalized counts.

    Returns:
        DataFrame of normalized (and optionally log-transformed) counts.
    """
    logger.info("Performing Data Normalization")

    if batch_col:     
        logging.info("Batch column input: {batch_col}")        
        logging.info("Performing batch correction")
        # pycombat expects (genes x samples)
        batch = metadata_df[batch_col].tolist()
        data_for_dds = pycombat_seq(counts_df, batch).T
    else:
        logging.info("No batch column input")
        logging.info("Normalization will be performed without batch correction")
        
        data_for_dds = counts_df.T

    inference = DefaultInference(n_cpus=1) 
    dds = DeseqDataSet(
        counts=pd.DataFrame(data_for_dds), # pd.df just to shut up pylance error
        metadata=metadata_df,
        design_factors='1',
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
        normed_counts = np.log1p(normed_counts)

    logger.info("Normalized data successfully")

    return normed_counts.T


def correlation_based_feature_selection(
    data: pd.DataFrame,
    acute_classification: pd.DataFrame,
    pval: float=0.05,
    cutoff_corr: float | None = None,
    cutoff_corr_genes: float | None = None,
    n_cpus: int = 8,
    verbose: bool=False
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
        pval: P-value threshold for significance when filtering gene-gene correlations.
        cutoff_corr: Minimum absolute Spearman R for including genes in the gene-gene graph.
        cutoff_corr_genes: Minimum absolute Pearson R for edge inclusion in the graph.

    Returns:
        DataFrame with selected genes and their correlation statistics.
    """

    _setup_internal_logger(verbose)

    logger.info("Initializing Correlation-Based Feature Learning")

    classified_samples_acute = acute_classification.loc[acute_classification['class'].notna(), 'sample']

    # Samples are columns
    dat_c = data.loc[:, classified_samples_acute]
    # logger.debug(f"\n{dat_c.to_markdown()}")

    corr1 = pd.DataFrame()
    
    logger.info("Performing correlation between Acute classification and genes expression")
    for i in range(0,dat_c.shape[0]):
        res = sc.spearmanr(dat_c.iloc[i,:], acute_classification.loc[:,'class'])
        res2 = pd.DataFrame.from_dict({'R': [res.statistic], 'p_value': [res.pvalue]})
        corr1 = pd.concat([corr1, res2])
    corr1.index = dat_c.index
    corr1['Genes'] = corr1.index
    logger.debug(f"corr1:\n{corr1.head(n=200).to_markdown()}")

    if not cutoff_corr:
        logger.info("Correlation between genes expression and the initial classification is not defined")
        logger.info("Using Q3 of absolute R values as cutoff")
        cutoff_corr = abs(corr1['R']).quantile(0.75)
    
    logger.info(f"R cutoff value between genes expression and classification: {cutoff_corr:.3f}")  
    corr_genes = corr1[(abs(corr1['R'])>=cutoff_corr) & (corr1['p_value']<=pval)].index.to_list()
    logger.info(f"Number of significant genes with R higher than {cutoff_corr:.3f}: {len(corr_genes)}")

    if len(corr_genes) == 0:

        logger.warning(f"There are no genes higher than {cutoff_corr}.")
        logger.warning(f"The maximum R value is {np.round(np.max(abs(corr1['R'])),2)}.") 
        logger.warning(f"By default the analyses will be performed only with significant correlations (p-value <= 0.05)") 
    
        corr_genes = corr1[corr1['p_value']<=pval].index.to_list()

    subdat = data.loc[corr_genes,:]

    logger.debug(f"Selecction of genes data:\n{subdat.head(n=200).to_markdown()}")
    logger.debug(f"Shape: {subdat.shape}")
    
    logger.info("Performing Gene to Gene correlation for Feature Selection")

    

    # # Create generator with combinations
    # gene_combinations = combinations(range(len(subdat)), 2)
    # index_loop = ((subdat, i, j) for i, j in gene_combinations)

    # # Pass generator to pool processes
    # with Pool(processes=n_cpus) as pool:
    #     results = pool.map(_corr_loop, index_loop)
    
    # Concatenate results
    # corr2 = pd.concat(results)

    corr2 = _correlation_of_gene_tuples(subdat.T)
    # pval_correction_method = 'fdr_bh'
    pval_correction_method = 'bonferroni'
    logger.debug(f"Using {pval_correction_method} to correct p-values")
    _, corr2['p_adj'], _, _ = multipletests(corr2['p_value'], alpha=0.05, method=pval_correction_method)
    

    if logger.level == logging.DEBUG:
        corr1.to_csv("corr1.csv", index=False, header = True) # DEBUG
        corr2.to_csv("corr2.csv", index=False, header = True) # DEBUG
        # corr_2_2.to_csv("corr2_2.csv", index=False, header = True) # DEBUG
    logger.debug(f"correlations Gene to Gene Results:\n{corr2.head(n=200).to_markdown()}")
    
    # Using iGraph for gene selection. 
    # Correlation threshold is used to build a graph. 
    # We only select the most representative (higher R with CLASSIFICATION) gene from each set

    if not cutoff_corr_genes:
        percentile = 0.95
        logger.info("Correlation between genes expression is not defined")
        logger.info(f"Using percentile {percentile} of absolute R values as cutoff")
        cutoff_corr_genes = abs(corr2['R']).quantile(percentile)
        

    logger.info(f"R Gene x Gene cutoff value: {cutoff_corr_genes:.3f}") # NOTE: change wording

    gene_tuples = corr2[(corr2['p_adj']<pval) & (abs(corr2['R'])>cutoff_corr_genes)]\
        .loc[:, ['Gene1', "Gene2"]]\
        .itertuples(index=False) # Generator
    
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
        logger.debug(f"Cluster values: \n{values.to_markdown()}")
        selected_gene = abs(values['R']).idxmax()
        logger.debug(f"Gene with max absolute R value from Cluster {i}: {selected_gene}")
        selected_genes.append(selected_gene)
       
    corr1_final = corr1.loc[(corr1['Genes'].isin(selected_genes) & (corr1['p_value'] <= pval))] # is pval needed? 
    logger.info(f"Number of final genes selected: {len(corr1_final)}")
    
    return corr1_final['Genes'].to_list()


    
def run_catrans(
        raw_counts: pd.DataFrame,
        metadata: pd.DataFrame,
        acute_classification: pd.DataFrame,
        models: list[Any],
        batch_col: str | None = None, # TODO: Review if it could accept more than a single string (should perform dimension reduction if so) ALSO it should be possible to not perform correction by batch, only normalization.
        model_threshold: float = 0.8,
        n_cpus: int = 8,
        signature_genes: list[str] | None = None,
        verbose: bool = False
    ) -> Any:

    _setup_internal_logger(verbose=verbose)

    # TODO: Instert here Feature Selection logic

    if not signature_genes:
        logging.info("No signature genes were provided.")
        logging.info("Feature selection will be performed with default parameters.")
        
        signature_genes = correlation_based_feature_selection(
            data=raw_counts,
            acute_classification=acute_classification,
            n_cpus=n_cpus,
            verbose=verbose
        )
        
        if signature_genes is None:
            raise ValueError("Feature selection failed to produce a gene signature.")


    logger.info("Initializing CATranS algorithm")
    
    # Argument checking

    ## Assert acute_classification is a dataframe with sample and class # NOTE: I don't really like this, however for now it would be difficult to change
    try:
        acute_classification[['sample', 'class']]
    except KeyError:
        raise ValueError("acute_classification must be a pandas.DataFrame with 'sample' and 'class' columns.")

    ## Assert acute_classification also has the samples as index 
    acute_classification.set_index(
        pd.Index(acute_classification['sample'].to_list()),
        inplace=True
        ) # NOTE: This is horrible
    
    ## Assert models have .predict_proba function
    for model in models:
        if not hasattr(model, 'predict_proba'):
            raise AttributeError("All models must have a predict_proba method in order to run semi-supervised learning.")
    
    
    
    # First steps prior to iteration


    # ## Perform Differential Expression Analysis with the acute classification to stablish a baseline
    # ## NOTE: Is it possible this isn't needed?
    # degs, dea_results_df = _diffExpAnalysis(
    #     data=raw_counts,
    #     classification=acute_classification,
    #     n_cpus=n_cpus,
    #     design='class', # NOTE: I don't like the hardcoded 'class'.
    # )

    ## Normalize and batch correct data

    logger.info("Normalizing DataSet...")

    # TODO: Normalization logic. should be possible to not pass any batch column to perform normalization
    if batch_col:
        normalized_data = _get_normalized_batch_corrected(
            counts_df=raw_counts,
            metadata_df=metadata,
            batch_col=batch_col
        )

    else:
        normalized_data = _get_normalized_batch_corrected(
            counts_df=raw_counts,
            metadata_df=metadata
        )

    history, degs_list, dea_results_list = _semisupervised_model(
        data=normalized_data, 
        data_raw=raw_counts,
        signature=signature_genes,
        acute_classification=acute_classification,
        models=models,
        threshold=model_threshold
    )
    

    return history, degs_list, dea_results_list 
    # NOTE: I don't like this. Should return an Object with all data? Would be bit more structured
    # Okay, not even that. This should ONLY give away the resulting classification, no?
    # Well, you can insert it in an Object that gives it like Object.final_classification or something like that

    # Despite everything, if it works it works.
    # I also should review the logic in some functions, since I believe some things don't work.
    