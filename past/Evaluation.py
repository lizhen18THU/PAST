import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score

def svm_cross_validation(mtx, target, Kfold=5):
    """
    K-fold cross validation, taking low-dimensional embedding as input, annotation as output and SVM with rbf kernel as classifier

    Parameters
    ------
    mtx
        latent feature matrix
    target
        annotation
    Kfold
        number of fold for cross validation

    Returns
    ------
    Acc
        Accuracy of cross validation
    K
        Kappa of cross validation
    mF1
        mF1 of cross validation
    wF1
        wF1 of cross validation
    """
    if isinstance(target, pd.Series):
        target = target.astype(str).values

    target_unique = np.unique(target).reshape(1, -1)
    target_onehot = (target.reshape(-1, 1)==target_unique).astype(int)
    target = target_onehot.argmax(-1)
    svc = SVC()
    cv_results = cross_validate(svc, mtx, target,
                                scoring=("accuracy", "f1_macro", "f1_weighted"),
                                cv=Kfold, n_jobs=Kfold)
    svc = SVC()
    from sklearn.metrics import cohen_kappa_score, make_scorer
    kappa_score = make_scorer(cohen_kappa_score)
    kappa = cross_validate(svc, mtx, target,
                                scoring=kappa_score,
                                cv=Kfold, n_jobs=Kfold)["test_score"]
    
    return cv_results["test_accuracy"], kappa, cv_results["test_f1_macro"], cv_results["test_f1_weighted"]

def cluster_refine(pred, spatial_mtx, num_nbs=6):
    """
    Refine clustering result according spatial neighborhood

    Parameters
    ------
    pred
        original clustering labels
    spatial_mtx
        spatial cordinate matrix
    num_nbs
        number of neighbors to consider when refining clustering labels

    Returns
    ------
    refined_label
        refined labels
    """

    assert num_nbs > 0, "The number of neighbors must be larger than zero"
    sample_id = [i for i in range(pred.shape[0])]
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = kneighbors_graph(spatial_mtx, n_neighbors=num_nbs).toarray()
    dis_df=pd.DataFrame(dis_df, index=sample_id, columns=sample_id)
    for i in range(len(sample_id)):
        index=sample_id[i]
        nbs = dis_df.loc[index, :]
        nbs_pred=pred.loc[nbs.index[nbs.values>0], "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    print("Finish refining")
    return refined_pred

def default_louvain(adata, refine=False, num_nbs=6, use_rep="embedding"):
    """
    Default louvain clustering algorithm applied in scanpy package with default resolution 1.0

    Parameters
    ------
    adata
        target dataset of anndata format with latent feature stored in adata.obsm[use_rep] or with result of pp.neighbors
    refine
        whether or not refine clustering results, if True, spatial coordinate should be stored in adata.obsm["spatial"]
    num_nbs
        number of neighbors to consider when refining clustering labels, valid only if refine is True
    use_rep
        key of adata.obsm implying latent features

    Returns
    ------
    adata
        target dataset of anndata format with default louvain clustering result stored in adata.obs["Dlouvain"] and
        refined clustering result stored in adata.obs["Dlouvain_refined"] if refined is True
    """

    if "neighbors" not in adata.uns.keys():
        sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.louvain(adata)
    adata.obs["Dlouvain"] = adata.obs["louvain"]
    if refine:
        adata.obs["Dlouvain_refined"] = cluster_refine(adata.obs["Dlouvain"].values, adata.obsm["spatial"], num_nbs)
    return adata

def default_leiden(adata, refine=False, num_nbs=6, use_rep="embedding"):
    """
    Default leiden clustering algorithm applied in scanpy package with default resolution 1.0

    Parameters
    ------
    adata
        target dataset of anndata format with latent feature stored in adata.obsm[use_rep] or with result of pp.neighbors
    refine
        whether or not refine clustering results, if True, spatial coordinate should be stored in adata.obsm["spatial"]
    num_nbs
        number of neighbors to consider when refining clustering labels, valid only if refine is True
    use_rep
        key of adata.obsm implying latent features

    Returns
    ------
    adata
        target dataset of anndata format with default leiden clustering result stored in adata.obs["Dleiden"] and
        refined clustering result stored in adata.obs["Dleiden_refined"] if refined is True
    """

    if "neighbors" not in adata.uns.keys():
        sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.leiden(adata)
    adata.obs["Dleiden"] = adata.obs["leiden"]
    if refine:
        adata.obs["Dleiden_refined"] = cluster_refine(adata.obs["Dleiden"].values, adata.obsm["spatial"], num_nbs)
    return adata

def run_louvain(adata, n_cluster, refine=False, num_nbs=6, use_rep="embedding", range_min=0, range_max=3, max_steps=30):
    """
    Search resolution so that louvain clustering algorithm obtain cluster numbers as close to given number as possible

    Parameters
    ------
    adata
        target dataset of anndata format with latent feature stored in adata.obsm[use_rep] or with result of pp.neighbors
    n_cluster
        cluster numbers
    refine
        whether or not refine clustering results, if True, spatial coordinate should be stored in adata.obsm["spatial"]
    num_nbs
        number of neighbors to consider when refining clustering labels, valid only if refine is True
    use_rep
        key of adata.obsm implying latent features
    range_min
        start resolution to search
    range_max
        end  resolution  to search
    max_steps
        max iterators to search resolution

    Returns
    ------
    adata
        target dataset of anndata format with searched louvain clustering result stored in adata.obs["Nlouvain"] and
        refined clustering result stored in adata.obs["Nlouvain_refined"] if refined is True
    """

    if "neighbors" not in adata.uns.keys():
        sc.pp.neighbors(adata, use_rep=use_rep)
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        this_resolution = this_min + ((this_max-this_min)/2)
        sc.tl.louvain(adata, resolution=this_resolution)
        this_clusters = adata.obs['louvain'].nunique()

        if this_clusters > n_cluster:
            this_max = this_resolution
        elif this_clusters < n_cluster:
            this_min = this_resolution
        else:
            print("Succeed to find %d clusters at resolution %.3f"%(n_cluster, this_resolution))
            adata.obs["Nlouvain"] = adata.obs["louvain"]
            if refine:
                adata.obs["Nlouvain_refined"] = cluster_refine(adata.obs["Nlouvain"].values, adata.obsm["spatial"], num_nbs)
            return adata
        this_step += 1

    print('Cannot find the number of clusters')
    adata.obs["Nlouvain"] = adata.obs["louvain"]
    if refine:
        adata.obs["Nlouvain_refined"] = cluster_refine(adata.obs["Nlouvain"].values, adata.obsm["spatial"], num_nbs)
    return adata


def run_leiden(adata, n_cluster, refine=False, num_nbs=6, use_rep="STAGATE", range_min=0, range_max=3, max_steps=30):
    """
    Search resolution so that leiden clustering algorithm obtain cluster numbers as close to given number as possible

    Parameters
    ------
    adata
        target dataset of anndata format with latent feature stored in adata.obsm[use_rep] or with result of pp.neighbors
    n_cluster
        cluster numbers
    refine
        whether or not refine clustering results, if True, spatial coordinate should be stored in adata.obsm["spatial"]
    num_nbs
        number of neighbors to consider when refining clustering labels, valid only if refine is True
    use_rep
        key of adata.obsm implying latent features
    range_min
        start resolution to search
    range_max
        end  resolution  to search
    max_steps
        max iterators to search resolution

    Returns
    ------
    adata
        target dataset of anndata format with searched leiden clustering result stored in adata.obs["Nleiden"] and
        refined clustering result stored in adata.obs["Nleiden_refined"] if refined is True
    """

    if "neighbors" not in adata.uns.keys():
        sc.pp.neighbors(adata, use_rep=use_rep)
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        this_resolution = this_min + ((this_max - this_min) / 2)
        sc.tl.leiden(adata, resolution=this_resolution)
        this_clusters = adata.obs['leiden'].nunique()

        if this_clusters > n_cluster:
            this_max = this_resolution
        elif this_clusters < n_cluster:
            this_min = this_resolution
        else:
            print("Succeed to find %d clusters at resolution %.3f" % (n_cluster, this_resolution))
            adata.obs["Nleiden"] = adata.obs["leiden"]
            if refine:
                adata.obs["Nleiden_refined"] = cluster_refine(adata.obs["Nleiden"].values, adata.obsm["spatial"],
                                                              num_nbs)
            return adata
        this_step += 1

    print('Cannot find the number of clusters')
    adata.obs["Nleiden"] = adata.obs["leiden"]
    if refine:
        adata.obs["Nleiden_refined"] = cluster_refine(adata.obs["Nleiden"].values, adata.obsm["spatial"], num_nbs)
    return adata


def mclust_R(adata, num_cluster, refine=False, num_nbs=6, modelNames='EEE', used_obsm='embedding', random_seed=666):
    """
    Clustering using the mclust algorithm.

    Parameters
    ------
    adata
        target dataset of anndata format with latent feature stored in adata.obsm[used_obsm]
    num_cluster
        cluster number
    refine
        whether or not refine clustering results, if True, spatial coordinate should be stored in adata.obsm["spatial"]
    num_nbs
        number of neighbors to consider when refining clustering labels, valid only if refine is True
    modelNames
        parameter in mclust R package, implying different data distribution
    used_obsm
        key of adata.obsm implying latent features
    random_seed
        seed for reproduction

    Returns
    ------
    adata
        target dataset of anndata format with mclust clustering result stored in adata.obs["mclust"] and
        refined clustering result stored in adata.obs["mclust_refined"] if refined is True
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    embedding = (adata.obsm[used_obsm] - adata.obsm[used_obsm].mean(axis=0)) / adata.obsm[used_obsm].std(axis=0)
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(embedding), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    if refine:
        adata.obs["mclust_refined"] = cluster_refine(adata.obs["mclust"].values, adata.obsm["spatial"], num_nbs)
        adata.obs["mclust_refined"] = adata.obs["mclust_refined"].astype("category")

    return adata

def cluster_metrics(adata, target, pred):
    """
    clustering metrics including ARI, AMI, HOMO and NMI

    Parameters
    ------
    adata
        target dataset of anndata format with target key and pred key
    target
        key stored in adata.obs implying ground truth
    pred
        key stored in adata.obs implying clustering result

    Returns
    ------
    ari
        adjusted rand index
    ami
        adjusted mutual information score
    nmi
        normalized mutual information score
    fmi
        fowlkes–mallows index
    comp
        completeness score
    homo
        homogeneity score
    """
    vec_target = adata.obs[target].astype(str)
    vec_pred = adata.obs[target].astype(str)
    
    ari = adjusted_rand_score(vec_target, vec_pred)
    nmi = normalized_mutual_info_score(vec_target, vec_pred)
    ami = adjusted_mutual_info_score(vec_target, vec_pred)
    comp = completeness_score(vec_target, vec_pred)
    fmi = fowlkes_mallows_score(vec_target, vec_pred)
    homo = homogeneity_score(vec_target, vec_pred)

    return ari, ami, nmi, fmi, comp, homo
