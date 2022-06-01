import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import jit
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import Dataset, DataLoader
import torch

def setup_seed(seed):
    """
    Set random seed
    :param seed: Number to be set as random seed for reproducibility.
    :return: None
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed_all(seed)  #gpus
    torch.backends.cudnn.deterministic = True


def DLPFC_split(adata, dataset_key, anno_key, percentage=1.0, dataset_filter=[None]):
    assert percentage>=0 and percentage<=1
    adata_final = None
    if not isinstance(dataset_filter, list):
        dataset_filter = list(dataset_filter)
    datasets = list(np.unique(adata.obs[dataset_key]))
    datasets = list(set(datasets) - set(dataset_filter))
    datasets.sort()
    print(datasets)
    for dataset in datasets:
        adata_dataset = adata[adata.obs[dataset_key]==dataset, :]
        labels = np.unique(adata_dataset.obs[anno_key])
        print(labels)
        for label in labels:
            adata_cur = adata_dataset[adata_dataset.obs[anno_key]==label, :]
#             print(adata_cur)
            if adata_final is None:
                adata_final = adata_cur[0:int(adata_cur.X.shape[0]*percentage), :]
            else:
                adata_final = sc.AnnData.concatenate(adata_final, adata_cur[0:int(adata_cur.X.shape[0]*percentage), :])
#             print(adata_final)
    return adata_final

def integration(adata_ref, adata):
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
    adata_ref = sc.AnnData(sp.csr_matrix(express_df.values), obs=adata_ref.obs, var=pd.DataFrame(target_features))

    return adata_ref


def geary_genes(adata, n_tops=3000, k=30):
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


def preprocess(adata, min_cells=3, target_sum=None, is_filter_MT=True, n_tops=None, gene_method="hvg", normalize=True):
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    if min_cells is not None and min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    if is_filter_MT:
        # filter MT-gene
        adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        adata = adata[:, adata.var_names.str.startswith('MT-') == False]
    if n_tops is not None and n_tops < adata.X.shape[1] and gene_method.lower() == "hvg":
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_tops, subset=True)
    if normalize:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
    if n_tops is not None and n_tops < adata.X.shape[1] and gene_method.lower() == "gearyc":
        adata = geary_genes(adata, n_tops=n_tops)

    return adata


def get_bulk(adata_ref, key, min_samples=11, r=0.5):
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
    if not isinstance(graph, sp.coo_matrix):
        graph = sp.coo_matrix(graph)
    num_nodes = graph.shape[0]
    number_subgraph = (num_nodes * total_times) // batchsize + 1

    if batchsize >= num_nodes:
        print("This dataset is small enough to train without Ripplewalk_sampler")
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


def optim_parameters(net, included=None, excluded=None):
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
    dist = kneighbors_graph(feature_matrix, n_neighbors=k_neighbors, mode="distance")
    dist.data = (1 + 1e-6 + dist.data ** 2 / v) ** (-(1 + v) / 2)
    spatial_graph = sp.csr_matrix(dist / dist.sum(-1))

    return spatial_graph

def load_noise(sdata, mask):
    if sp.issparse(sdata.X):
        sdata.X = sdata.X.toarray()
    sdata.X = sdata.X * mask
    if sp.issparse(sdata.X):
        sdata.X = sp.csr_matrix
    return sdata.copy()