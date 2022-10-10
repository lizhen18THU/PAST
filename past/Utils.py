import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import jit
import scanpy as sc
from sklearn.svm import SVC
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import Dataset
import torch


def setup_seed(seed):
    """
    Set random seed

    Parameters
    ------
    seed
        Number to be set as random seed for reproducibility

    Returns
    ------
    None
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpus
    torch.backends.cudnn.deterministic = True


def DLPFC_split(adata, dataset_key, anno_key, percentage=1.0, dataset_filter=[None]):
    """
    Based on domain annotation and dataset annotation, apply Stratified downsampling to DLPFC dataset

    Parameters
    ------
    adata
        Whole DLPFC containing 12 sub-datasets of anndata format
    dataset_key
        Key stored in adata.obs presenting different sub-datasets
    anno_key
        Key stored in adata.obs presenting domain annotation
    percentage
        Downsampling percentage
    dataset_filter
        Sub-datasets to be filtered

    Returns
    ------
    adata_final
        Downsampled DLPFC of anndata format
    """

    assert percentage >= 0 and percentage <= 1
    adata_final = None
    if not isinstance(dataset_filter, list):
        dataset_filter = list(dataset_filter)
    datasets = list(np.unique(adata.obs[dataset_key]))
    datasets = list(set(datasets) - set(dataset_filter))
    datasets.sort()
    print(datasets)
    for dataset in datasets:
        adata_dataset = adata[adata.obs[dataset_key] == dataset, :]
        labels = np.unique(adata_dataset.obs[anno_key])
        print(labels)
        for label in labels:
            adata_cur = adata_dataset[adata_dataset.obs[anno_key] == label, :]
            if adata_final is None:
                adata_final = adata_cur[0:int(adata_cur.X.shape[0] * percentage), :]
            else:
                adata_final = sc.AnnData.concatenate(adata_final,
                                                     adata_cur[0:int(adata_cur.X.shape[0] * percentage), :])
    return adata_final


def integration(adata_ref, adata):
    """
    Align the gene set of reference dataset with that of target dataset

    Parameters
    ------
    adata_ref
        reference dataset of anndata format
    adata
        target dataset of anndata format

    Returns
    ------
    adata_ref
        reference dataset aligned with target dataset
    """

    def gene_union(express_df, target_features):
        target_features = list(map(lambda x: x.lower(), target_features))
        express_df.columns = list(map(lambda x: x.lower(), express_df.columns))
        features1 = list(set.intersection(set(express_df.columns), set(target_features)))
        features2 = list(set(target_features) - set(express_df.columns))
        zero_df = pd.DataFrame(np.zeros((express_df.shape[0], len(features2))), index=express_df.index,
                               columns=features2)
        express_df = pd.concat([express_df.loc[:, features1], zero_df], axis=1)
        print("add %d zero features to reference; Current total %d genes" % (len(features2), express_df.shape[1]))
        return express_df.loc[:, target_features]

    express_df = pd.DataFrame(adata_ref.X.toarray(), index=adata_ref.obs_names, columns=adata_ref.var_names)
    target_features = adata.var_names
    express_df = gene_union(express_df, target_features)

    adata_ref_obsm = adata_ref.obsm
    adata_ref_uns = adata_ref.uns
    adata_ref = sc.AnnData(sp.csr_matrix(express_df.values), obs=adata_ref.obs.copy(),
                           var = pd.DataFrame(index=target_features))
    adata_ref.obsm = adata_ref_obsm
    adata_ref.uns = adata_ref_uns

    return adata_ref


def geary_genes(adata, n_tops=3000, k=30):
    """
    Select spatially variable genes for better downstream analysis

    Parameters
    ------
    adata
        target dataset of anndata format
    n_tops
        number of spatially variable genes to select
    k
        number of neighbors to consider when construct K-NN graph

    Returns
    ------
    adata
        target dataset of anndata format containing n_top SVGs
    """

    @jit(nopython=True, parallel=True)
    def _geary(exp_mtx, data, row, col):
        N = exp_mtx.shape[0]
        sum_W = np.sum(data)
        geary_index = []
        for i in range(exp_mtx.shape[1]):
            exp_vec = exp_mtx[:, i]
            C = (N - 1) * (data * ((exp_vec[row] - exp_vec[col]) ** 2)).sum() / (
                    2 * sum_W * np.sum((exp_vec - exp_vec.mean()) ** 2))
            geary_index.append(1 - C)
        return geary_index

    W = kneighbors_graph(adata.obsm["spatial"], n_neighbors=k, include_self=False).tocoo()
    exp_mtx = adata.X.toarray() if sp.issparse(adata.X) else adata.X.copy()

    # calculate geary's c index
    geary_index = _geary(exp_mtx, W.data, W.row, W.col)

    # select n_top genes with highest score
    genes_selected = np.array([False] * adata.shape[1])
    genes_selected[np.argpartition(geary_index, len(geary_index) - n_tops)[-n_tops:]] = True

    return adata[:, genes_selected]


def preprocess(adata, min_cells=3, target_sum=None, is_filter_MT=False, n_tops=None, gene_method="hvg", normalize=True):
    """
    Data preprocess for downstream analysis

    Parameters
    ------
    adata
        target dataset of anndata format to be preprocessed
    min_cells
        number of cells in which each gene should as least express
    target_sum
        total gene expression to normalize each cell
    is_filter_MT
        whether or not to filter mitochondrial genes
    n_tops
        number of SVGs or HVGs to select, if None then keep all genes
    gene_method
        strategy to select genes, if 'hvg' then use 'seurat_v3' method applied in scanpy package to select HVGs, else if 'gearyc'
        then use geary's c statistics to select SVGs, else keep all genes
    normalize
        whether or not to normalize gene expression matirx

    Returns
    ------
    adata
        target dataset of anndata format after preprocessing
    """

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    if min_cells is not None and min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    if is_filter_MT:
        # filter MT-gene
        adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-') | adata.var_names.str.startswith('mT-') | adata.var_names.str.startswith('Mt-')
        adata = adata[:, adata.var['mt'] == False]
    if n_tops is not None and n_tops < adata.X.shape[1] and gene_method.lower() == "hvg":
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_tops, subset=True)
    if normalize:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
    if n_tops is not None and n_tops < adata.X.shape[1] and gene_method.lower() == "gearyc":
        adata = geary_genes(adata, n_tops=n_tops)

    return adata


def get_bulk(adata_ref, key, min_samples=11, r=0.5):
    """
    construct pseudo bulk from reference dataset according to annotation

    Parameters
    ------
    adata_ref
        reference dataset of anndata format
    key
        key of the annoatation
    min_samples
        minimum number of pseudo bulk samples should be constructed from reference dataset
    r
        ratio for sampling from reference dataset to construct pseudo bulk sample

    Returns
    ------
    adata_bulk
        pseudo bulk samples of anndata format
    """

    index_list = []
    key_index = adata_ref.obs[key].values
    bulks = np.unique(key_index)
    df_index = []
    if min_samples is not None and min_samples > bulks.shape[0]:
        times = min_samples // bulks.shape[0]
        print("DownSample %d times to get enough pseudo bulks!" % times)
        for item in bulks:
            cur_index = np.argwhere(key_index == item).reshape(-1)
            index_list.append(cur_index)

            length = cur_index.shape[0]
            for i in range(times):
                shuffle_index = cur_index.copy()
                np.random.shuffle(shuffle_index)
                index_list.append(shuffle_index[0:max(1, int(length * r))])
    else:
        for item in bulks:
            cur_index = np.argwhere(key_index == item).reshape(-1)
            index_list.append(cur_index)
            df_index.append(item)

    for i, index in enumerate(index_list):
        if i == 0:
            data_mtx = np.average(adata_ref[index, :].X.toarray(), axis=0).reshape(1, -1)
        else:
            data_mtx = np.concatenate([data_mtx, np.average(adata_ref[index, :].X.toarray(), axis=0).reshape(1, -1)],
                                      axis=0)
    print("bulk_data's shape:", data_mtx.shape)
    adata_bulk = sc.AnnData(data_mtx, var=adata_ref.var.copy())

    return adata_bulk


def visualize(adata, keys, use_rep="embedding", library_id=None, spot_size=None, plot_spatial=True):
    """
    visualization for cell embedding and spatial clustering

    Parameters
    ------
    adata
        target dataset of anndata format
    keys
        keys to visualize stored in adata.obs
    use_rep
        embedding storing in adata.obsm for visualizing
    library_id
        visualize spatial clustering on top of tissue histology, specific to 10X Visium
    spot_size
        spot size for spatial clustering visualization
    plot_spatial
        whether or not to visualize spatial clustering

    Returns
    ------
    None
    """

    assert isinstance(keys, list)
    if "neighbors" not in adata.uns.keys():
        sc.pp.neighbors(adata, use_rep=use_rep)
    if "X_umap" not in adata.obsm.keys():
        sc.tl.umap(adata)
    for key in keys:
        sc.pl.umap(adata, color=key)
        if plot_spatial:
            if library_id is not None:
                sc.pl.spatial(adata, color=key, library_id=library_id)
            elif spot_size is not None:
                sc.pl.spatial(adata, color=key, spot_size=spot_size)
            else:
                sc.pl.embedding(adata, color=key, basis="spatial")


class StDataset(Dataset):
    """
    Spatial Transcriptomic Dataset
    
    Parameters
    ------
    data
        gene expression of the dataset
    knn
        K-NN graph constructed according the spatial coordinate of dataset
    metric
        spatial prior graph constructed according the spatial coordinate of dataset
    """

    def __init__(self, data, knn, metric):
        self.data = data
        self.knn = knn
        self.metric = metric

    def __getitem__(self, index):
        data = self.data[index]
        knn = self.knn[index]
        metric = self.metric[index]

        return (data, knn, metric)

    def __len__(self):
        return self.data.shape[0]

    def get_batch(self, index):
        """
        Get a batch of data according to given index
        
        Parameters
        ------
        index
            a list of index for a batch of samples
             
        Returns
        ------
        data_batch
            gene expression for a batch of data
        knn_batch
            K-NN sub-graph for a batch of data
        metric_batch
            spatial prior sub-graph for a batch of data
        """

        data_batch = self.data[index]
        knn_batch = self.knn[index][:, index]
        metric_batch = self.metric[index][:, index]

        if sp.issparse(data_batch):
            data_batch = data_batch.toarray()
        if not isinstance(data_batch, torch.FloatTensor):
            data_batch = torch.FloatTensor(data_batch)

        if sp.issparse(knn_batch):
            knn_batch = knn_batch.toarray()
        if not isinstance(knn_batch, torch.FloatTensor):
            knn_batch = torch.FloatTensor(knn_batch)

        if sp.issparse(metric_batch):
            metric_batch = metric_batch.toarray()
        if not isinstance(metric_batch, torch.FloatTensor):
            metric_batch = torch.FloatTensor(metric_batch)

        return (data_batch, knn_batch, metric_batch)


def Ripplewalk_sampler(graph, r=0.5, batchsize=6400, total_times=10):
    """
    Training stategy of subgraph segmentation based on random walk, enabling mini-batch training on large datasets
    
    Parameters
    ------
    graph
        graph indicating connectivity between spots, usually K-NN graph constructed from spatial coordinates
    r
        expansion ratio for sampling subgraph
    batchsize
        number of samples for a mini-batch
    total_times
        decide the number of subgraph to sample, number of subgraph = total_times * dataset_size / batchsize
        
    Returns
    ------
    subgraph_set
        a list containing index of sampled sub-graphs
    """

    if not isinstance(graph, sp.coo_matrix):
        graph = sp.coo_matrix(graph)
    num_nodes = graph.shape[0]
    number_subgraph = (num_nodes * total_times) // batchsize + 1

    if batchsize >= num_nodes:
        print("This dataset is smaller than batchsize so that ripple walk sampler is not used!")
        subgraph_set = []
        for i in range(total_times):
            index_list = [j for j in range(num_nodes)]
            random.shuffle(index_list)
            subgraph_set.append(index_list)
        return subgraph_set

    # transform adj to index
    final = []
    for i in range(num_nodes):
        final.append(graph.col[graph.row == i].tolist())
    graph = final

    # Ripplewalk sampling
    subgraph_set = []
    for i in range(number_subgraph):
        # select initial node, and store it in the index_subgraph list
        index_subgraph = [np.random.randint(0, num_nodes)]
        # the neighbor node set of the initial nodes
        neighbors = graph[index_subgraph[0]]
        len_subgraph = 1
        while (1):
            len_neighbors = len(neighbors)
            if (len_neighbors == 0):  # getting stuck in the inconnected graph, select restart node
                while (1):
                    restart_node = np.random.randint(0, num_nodes)
                    if (restart_node not in index_subgraph):
                        break
                index_subgraph.append(restart_node)
                neighbors = neighbors + graph[restart_node]
                neighbors = list(set(neighbors) - set(index_subgraph))
                len_subgraph = len(index_subgraph)
            else:
                # select part (half) of the neighbor nodes and insert them into the current subgraph
                if ((batchsize - len_subgraph) > (len_neighbors * r)):  # judge if we need to select that much neighbors
                    neig_random = random.sample(neighbors, max(1, int(r * len_neighbors)))
                    neighbors = list(set(neighbors) - set(neig_random))

                    index_subgraph = index_subgraph + neig_random
                    index_subgraph = list(set(index_subgraph))
                    for i in neig_random:
                        neighbors = neighbors + graph[i]
                    neighbors = list(set(neighbors) - set(index_subgraph))
                    len_subgraph = len(index_subgraph)
                else:
                    neig_random = random.sample(neighbors, (batchsize - len_subgraph))
                    index_subgraph = index_subgraph + neig_random
                    index_subgraph = list(set(index_subgraph))
                    break
        subgraph_set.append(index_subgraph)
    return subgraph_set


def Ripplewalk_prediction(graph, r=0.5, batchsize=6400):
    """
    Prediction stategy of subgraph segmentation based on random walk, enabling mini-batch prediction on large datasets
    
    Parameters
    ------
    graph
        graph indicating connectivity between spots, usually K-NN graph constructed from spatial coordinates
    r
        expansion ratio for sampling subgraph
    batchsize
        number of samples for a mini-batch
        
    Returns
    ------
    subgraph_set
        a list containing index of sampled sub-graphs
    """
        
    if not isinstance(graph, sp.coo_matrix):
        graph = sp.coo_matrix(graph)
    num_nodes = graph.shape[0]
    
    if batchsize >= num_nodes:
        subgraph_set = []
        index_list = np.array([j for j in range(num_nodes)])
        subgraph_set.append(index_list)
        return subgraph_set
    
    # transform adj to index
    final = []
    for i in range(num_nodes):
        final.append(graph.col[graph.row==i].tolist())
    graph = final
    
    all_nodes = [j for j in range(num_nodes)]
    sampled_nodes = []
    
    # Ripplewalk sampling
    subgraph_set = []
    
    while len(all_nodes)>len(sampled_nodes):
        # select initial node from non-sampled nodes
        nonsampled_nodes = np.setdiff1d(all_nodes, sampled_nodes)
        random.shuffle(nonsampled_nodes)
        cur_node = nonsampled_nodes[0]
        index_subgraph = [cur_node]
        
        #the neighbor node set of the initial nodes
        neighbors = graph[index_subgraph[0]]
        sampled_nodes.append(cur_node)
        len_subgraph = 1
        
        while(1):
            len_neighbors = len(neighbors)
            if(len_neighbors == 0): # getting stuck in the inconnected graph, select restart node
                nonsampled_nodes = np.setdiff1d(all_nodes, sampled_nodes)
                random.shuffle(nonsampled_nodes)
                restart_node = nonsampled_nodes[0]
                index_subgraph.append(restart_node)
                
                neighbors = neighbors + graph[restart_node]
                neighbors = list(set(neighbors) - set(index_subgraph))
                sampled_nodes.append(restart_node)
                len_subgraph = len(index_subgraph)
            else: # select part (half) of the neighbor nodes and insert them into the current subgraph
                if ((batchsize - len_subgraph) > (len_neighbors*r)): # judge if we need to select that much neighbors
                    neig_random = random.sample(neighbors, max(1, int(r*len_neighbors)))
                    neighbors = list(set(neighbors) - set(neig_random))

                    index_subgraph = index_subgraph + neig_random
                    index_subgraph = list(set(index_subgraph))
                    
                    for i in neig_random:
                        neighbors = neighbors + graph[i]
                    neighbors = list(set(neighbors) - set(index_subgraph))
                    sampled_nodes = np.union1d(sampled_nodes, neig_random).tolist()
                    len_subgraph = len(index_subgraph)
                else:
                    neig_random = random.sample(neighbors, (batchsize - len_subgraph))
                    index_subgraph = index_subgraph + neig_random
                    index_subgraph = list(set(index_subgraph))
                    sampled_nodes = np.union1d(sampled_nodes, neig_random).tolist()
                    break
        subgraph_set.append(np.array(index_subgraph))
    return subgraph_set


def optim_parameters(net, included=None, excluded=None):
    """
    Parameters in neural network to be trained
    
    Parameters
    ------
    net
        neural network model
    included
        parameters to be trained, higher priority to excluded
    excluded
        parameters fixed, lower priority to excluded
    
    Returns
    ------
    params
        iterator of parameter set to be trained
    """

    def belongs_to(cur_layer, layers):
        for layer in layers:
            if layer in cur_layer:
                return True
        return False

    params = []
    if included is not None:
        if not isinstance(included, list):
            included = [included]
        for cur_layer, param in net.named_parameters():
            if belongs_to(cur_layer, included) and param.requires_grad:
                params.append(param)
    else:
        if not isinstance(excluded, list):
            excluded = [excluded] if excluded is not None else []
        for cur_layer, param in net.named_parameters():
            if not belongs_to(cur_layer, excluded) and param.requires_grad:
                params.append(param)

    return iter(params)


def spatial_prior_graph(feature_matrix, k_neighbors, v=1):
    """
    Construct spatial prior graph for metric learning
    
    Parameters
    ------
    feature_matrix
        spatial corrdinate matrix
    k_neighbors
        number of neighbors to construct graph
    v
        scale factor in student's t kernel
    
    Returns
    ------
    spatial_graph
        spatial graph of sparse matrix format
    """
    dist = kneighbors_graph(feature_matrix, n_neighbors=k_neighbors, mode="distance").tocoo()
    dist.data = (1 + 1e-6 + dist.data**2/v) ** (-(1+v) / 2)
    dist.data = dist.data/(np.array(dist.sum(axis=1)).reshape(-1, 1).repeat(k_neighbors, axis=1).reshape(-1))
    spatial_graph = dist.tocsr()
    
    return spatial_graph


def load_noise(sdata, mask):
    """
    Add noise to gene expression matrix
    
    Parameters
    ------
    sdata
        target dataset of anndata format
    mask
        matrix containing 0 or 1 to mask gene expression as noise
    
    Returns
    ------
    sdata
        target dataset of anndata format with noise
    """

    if sp.issparse(sdata.X):
        sdata.X = sdata.X.toarray()
    sdata.X = sdata.X * mask
    if sp.issparse(sdata.X):
        sdata.X = sp.csr_matrix
    return sdata.copy()

def svm_annotation(ref_mtx, ref_anno, target_mtx):
    """
    Use SVM with radial basis function kernel as classifier to train on the reference dataset and annotate the target dataset.

    Parameters
    ------
    ref_mtx
        feature matrix of reference dataset, usually referring to latent embedding of PAST
    ref_anno
        the annotation of reference dataset
    target_mtx
        feature matrix of target dataset in the same latent space with that of reference dataset

    Returns
    ------
        the annotation of target dataset
    """

    if isinstance(ref_anno, pd.Series):
        ref_anno = ref_anno.astype(str).values
    ref_anno_unique = np.unique(ref_anno).reshape(1, -1)
    ref_onehot = (ref_anno.reshape(-1, 1)==ref_anno_unique).astype(int)
    ref_out = ref_onehot.argmax(-1)
    ref_anno_unique = ref_anno_unique.reshape(-1)
    
    svc = SVC()
    svc.fit(ref_mtx, ref_out)
    target_out = svc.predict(target_mtx)
    
    return ref_anno_unique[target_out]