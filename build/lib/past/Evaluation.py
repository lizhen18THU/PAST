import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import normalized_mutual_info_score

def svm_cross_validation(mtx, target, Kfold=5):
    """
    Take low-dimensional embedding as input, annotation as output, SVM with rbf kernel as classifier,
    """
    if isinstance(target, pd.Series):
        target = target.astype(str).values

    target_unique = np.unique(target).reshape(1, -1)
    target_onehot = (target.reshape(-1, 1) == target_unique).astype(int)
    target = target_onehot.argmax(-1)
    svc = SVC()
    cv_results = cross_validate(svc, mtx, target, scoring="accuracy", cv=Kfold, n_jobs=Kfold)

    return cv_results["test_score"]

def cluster_refine(pred, spatial_mtx, num_nbs=6):
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
    if "neighbors" not in adata.uns.keys():
        sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.louvain(adata)
    adata.obs["Dlouvain"] = adata.obs["louvain"]
    if refine:
        adata.obs["Dlouvain_refined"] = cluster_refine(adata.obs["Dlouvain"].values, adata.obsm["spatial"], num_nbs)
    return adata

def default_leiden(adata, refine=True, num_nbs=6, use_rep="embedding"):
    if "neighbors" not in adata.uns.keys():
        sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.leiden(adata)
    adata.obs["Dleiden"] = adata.obs["leiden"]
    if refine:
        adata.obs["Dleiden_refined"] = cluster_refine(adata.obs["Dleiden"].values, adata.obsm["spatial"], num_nbs)
    return adata

def run_louvain(adata, n_cluster, refine=True, num_nbs=6, use_rep="embedding", range_min=0, range_max=3, max_steps=30, tolerance=0):
    if "neighbors" not in adata.uns.keys():
        sc.pp.neighbors(adata, use_rep=use_rep)
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        this_resolution = this_min + ((this_max-this_min)/2)
        sc.tl.louvain(adata, resolution=this_resolution)
        this_clusters = adata.obs['louvain'].nunique()

        if this_clusters > n_cluster+tolerance:
            this_max = this_resolution
        elif this_clusters < n_cluster-tolerance:
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


def run_leiden(adata, n_cluster, refine=True, num_nbs=6, use_rep="STAGATE", range_min=0, range_max=3, max_steps=30,
               tolerance=0):
    if "neighbors" not in adata.uns.keys():
        sc.pp.neighbors(adata, use_rep=use_rep)
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        this_resolution = this_min + ((this_max - this_min) / 2)
        sc.tl.leiden(adata, resolution=this_resolution)
        this_clusters = adata.obs['leiden'].nunique()

        if this_clusters > n_cluster + tolerance:
            this_max = this_resolution
        elif this_clusters < n_cluster - tolerance:
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


def mclust_R(adata, num_cluster, refine=True, num_nbs=6, modelNames='EEE', used_obsm='embedding', random_seed=666):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
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
    ari = adjusted_rand_score(adata.obs[target], adata.obs[pred])
    ami = adjusted_mutual_info_score(adata.obs[target], adata.obs[pred])
    homo = homogeneity_score(adata.obs[target], adata.obs[pred])
    nmi = normalized_mutual_info_score(adata.obs[target], adata.obs[pred])
    print('ARI: %.3f, AMI: %.3f, Homo: %.3f, NMI: %.3f' % (ari, ami, homo, nmi))

    return ari, ami, homo, nmi