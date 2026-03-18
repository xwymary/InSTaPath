from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import pandas as pd
import scanpy as sc
import numpy as np
import anndata as ad
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from sklearn.decomposition import PCA

#%%
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2025):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter
    np.random.seed(random_seed)

    # load R package + seed
    ro.r("library(mclust)")
    ro.r["set.seed"](random_seed)

    X = np.asarray(adata.obsm[used_obsm], dtype=np.float64)  # (n,d)
    rmclust = ro.r["Mclust"]

    # Convert numpy -> R matrix inside a context (no activate())
    with localconverter(default_converter + numpy2ri.converter):
        rX = ro.conversion.py2rpy(X)

    # Call Mclust with named args (safer than positional)
    res = rmclust(rX, G=num_cluster, modelNames=modelNames)

    # Extract results by name (DON'T do res[-2])
    with localconverter(default_converter + numpy2ri.converter):
        labels = np.array(res.rx2("classification"))  # 1..K in R

    mclust_res = labels - 1  # optional: make it 0..K-1
    return mclust_res

#%%
def search_res(
        adata,
        n_clusters,
        method='leiden',
        use_rep='emb',
        start=0.01,
        end=2.0,
        increment=0.01):
    '''
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=False):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(
                pd.DataFrame(
                    adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(
                pd.DataFrame(
                    adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))

        if count_unique == n_clusters or count_unique > n_clusters:
            break

    return res


#%%
def clustering(
        adata,
        num_cluster,
        used_obsm,
        method,
        refine_cluster=False,
        n_neighbors=15,
        conf_proba=0.9,
        start=0.1,
        end=3,
        increment=0.02):

    if method == 'leiden':
        if isinstance(num_cluster, int):
            res = search_res(
                adata,
                num_cluster,
                use_rep=used_obsm,
                method=method,
                start=start,
                end=end,
                increment=increment)
        else:
            res = num_cluster
        sc.pp.neighbors(adata, n_neighbors=50, use_rep=used_obsm)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs[f"{used_obsm}_cluster"] = adata.obs['leiden']
    elif method == 'louvain':
        if isinstance(num_cluster, int):
            res = search_res(
                adata,
                num_cluster,
                use_rep=used_obsm,
                method=method,
                start=start,
                end=end,
                increment=increment)
        else:
            res = num_cluster
        sc.pp.neighbors(adata, n_neighbors=50, use_rep=used_obsm)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs[f"{used_obsm}_cluster"] = adata.obs['louvain']
    elif method == "kmeans":
        clusters = KMeans(
            n_clusters=num_cluster, n_init=100).fit(
            adata.obsm[used_obsm]).labels_
        adata.obs[f"{used_obsm}_cluster"] = clusters.astype(str)
    elif method == "bgm":
        bgm = BayesianGaussianMixture(
            n_components=num_cluster,
            init_params="random",
            covariance_type="full", # "tied"
            max_iter=1000,
            n_init=10,
            random_state=0
        ).fit(
            adata.obsm[used_obsm])
        clusters = bgm.predict(adata.obsm[used_obsm])
        adata.obs[f"{used_obsm}_cluster"] = clusters.astype(str)
        adata.obs[f"{used_obsm}_cluster_proba"] = bgm.predict_proba(
            adata.obsm[used_obsm]).max(axis=1)
    elif method == "mclust":
        mclust_res = mclust_R(adata, used_obsm=used_obsm, num_cluster=num_cluster)
        adata.obs[f"{used_obsm}_cluster"] = mclust_res.astype(str)
    if refine_cluster:
        if f"{used_obsm}_cluster_proba" in adata.obs:
            high_conf = adata.obs[f"{used_obsm}_cluster_proba"] > conf_proba # >0.9
            X = adata.obs[["array_row", "array_col"]].values[high_conf]
            y = adata.obs[f"{used_obsm}_cluster"].values[high_conf]
        else:
            X = adata.obs[["array_row", "array_col"]].values
            y = adata.obs[f"{used_obsm}_cluster"].values

        # make sure there are multiple clusters,
        # otherwise refinement doesn't make sense
        if np.unique(y).size > 1:
            neigh = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights="uniform",
                algorithm="brute")
            neigh.fit(X,
                      y)

            refined_clusters = neigh.predict(
                adata.obs[["array_row", "array_col"]].values)
            adata.obs[f"{used_obsm}_cluster"] = refined_clusters.astype(
                str)


#%%
def eval_metrics(X_embed, labels, method='mclust', pca_n_components=50, verbose=True, refine_cluster=False):
    adata = ad.AnnData(X=np.empty((X_embed.shape[0], 0)))
    adata.obsm['emb'] = X_embed
    adata.obs['Annotation'] = labels
    num_cluster = len(set(labels))
    
    if X_embed.shape[1]>50:
        print("Preprocess data with PCA.")
        pca = PCA(n_components=pca_n_components, random_state=42)
        embedding = pca.fit_transform(adata.obsm['emb'].copy())
        adata.obsm['emb'] = embedding
        clustering(adata, num_cluster=num_cluster, used_obsm='emb', method=method, refine_cluster=False)
    else:
        clustering(adata, num_cluster=num_cluster, used_obsm='emb', method=method, refine_cluster=False)
    ari = adjusted_rand_score(adata.obs['Annotation'], adata.obs['emb_cluster'])
    asw = silhouette_score(adata.obsm['emb'], adata.obs['Annotation'])
    nmi = normalized_mutual_info_score(adata.obs['Annotation'],  adata.obs['emb_cluster']) 
    if verbose: 
        print('ARI:', ari)
        print('ASW:', asw)
        print('NMI:', nmi)
    
    cluster = pd.to_numeric(adata.obs['emb_cluster'], errors="coerce").astype("Int64").astype(str).replace("<NA>", "NA").to_list()

    results = {'ari': ari, 'asw': asw, 'nmi': nmi, 'cluster': cluster}
    return results

#%%
def topic_coherence_npmi_numpy(X, topic_word, top_s=10, eps=1e-12, return_per_topic=False):
    """
    Topic Coherence using NPMI, pure NumPy version.

    Parameters
    ----------
    X : np.ndarray, shape (N_docs, V)
        Document-word (or document-code) matrix.
    topic_word : np.ndarray, shape (K, V)
        Topic-word probabilities.
    top_s : int
        Number of top words per topic.
    eps : float
        Small value to avoid log(0).
    return_per_topic : bool
        Whether to return per-topic coherence.

    Returns
    -------
    tc : float
        Global topic coherence.
    per_topic : np.ndarray (optional)
        Per-topic coherence.
    """

    X = np.asarray(X)
    topic_word = np.asarray(topic_word)

    N, V = X.shape
    K, V2 = topic_word.shape
    assert V == V2, "Vocabulary size mismatch."

    # Binary presence matrix
    X_bin = (X > 0).astype(int)

    # Marginal probabilities P(w)
    Pi = X_bin.sum(axis=0) / N   # shape (V,)

    per_topic = np.zeros(K)

    for k in range(K):
        # Top-s words for topic k
        top_idx = np.argsort(topic_word[k])[::-1][:top_s]
        s = len(top_idx)
        if s < 2:
            per_topic[k] = np.nan
            continue

        # Submatrix: N_docs x s
        sub = X_bin[:, top_idx]

        # Joint probabilities P(w_i, w_j)
        # (s x s) matrix
        Pij_mat = (sub.T @ sub) / N

        Pi_sub = Pi[top_idx]

        # Use only upper-triangle (i < j)
        iu, ju = np.triu_indices(s, k=1)
        Pij = Pij_mat[iu, ju]
        Pi_i = Pi_sub[iu]
        Pi_j = Pi_sub[ju]

        # Skip zero co-occurrence pairs
        mask = Pij > 0
        if not np.any(mask):
            per_topic[k] = np.nan
            continue

        Pij = Pij[mask]
        Pi_i = Pi_i[mask]
        Pi_j = Pi_j[mask]

        # NPMI
        numerator = np.log((Pij + eps) / ((Pi_i + eps) * (Pi_j + eps)))
        denominator = -np.log(Pij + eps)
        npmi = numerator / (denominator + eps)

        # Average NPMI for topic k
        per_topic[k] = np.mean(npmi)

    # Global TC
    tc = np.nanmean(per_topic)

    if return_per_topic:
        return tc, per_topic
    return tc

def calculate_td(beta, word_list, topics_list, num_top_words = 5):
    # beta [K,V]: Topic-word probabilities.
    df_beta = pd.DataFrame(beta.T, index = word_list, columns=topics_list)

    # find top 5 words per topic
    top_word_list = []
    for c in topics_list:
        top_word_list += df_beta[c].nlargest(num_top_words).index.tolist()
        
    K = len(topics_list)
    td = len(set(top_word_list))/(K*num_top_words)
    return td
