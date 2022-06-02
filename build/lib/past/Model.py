import torch
import time
import torch.nn as nn
from sklearn.neighbors import kneighbors_graph
from .Modules import *
from .Utils import *
from .Loss import *

## pca_fc + ScaledDotProductAttention + VAEplus
class RAST(nn.Module):
    def __init__(self, d_in, d_lat, k_neighbors=10, dropout=0.1):
        super().__init__()
        assert d_in > d_lat, "d_in should be larger than d_lat"
        assert k_neighbors > 0, "k_neighbors should be larger than 0"

        self.k_neighbors = k_neighbors
        self.d_prior = d_lat // 5
        self.enc_fc1 = BayesianLinear(d_in, self.d_prior)
        self.enc_fc2 = nn.Linear(d_in, d_lat - self.d_prior, bias=False)
        self.enc_attn1 = MaskedScaleDotProductAttention(d_in=d_lat, d_out=d_lat, d_k=d_lat, dropout=dropout)
        self.enc_attn2 = MaskedScaleDotProductAttention(d_in=d_lat, d_out=d_lat, d_k=d_lat, dropout=dropout)

        self.mu_fc = nn.Linear(d_lat, d_lat, bias=True)
        self.logvar_fc = nn.Linear(d_lat, d_lat, bias=True)

        self.dec_attn1 = MaskedScaleDotProductAttention(d_in=d_lat, d_out=d_lat, d_k=d_lat, dropout=dropout)
        self.dec_attn2 = MaskedScaleDotProductAttention(d_in=d_lat, d_out=d_lat, d_k=d_lat, dropout=dropout)
        self.dec_fc = nn.Linear(d_lat, d_in, bias=False)

    def prior_initialize(self, prior):
        if not isinstance(prior, torch.FloatTensor):
            prior = torch.FloatTensor(prior)
        assert prior.shape[0] == self.d_prior, "prior weight dimension not match"

        prior_log_sigma = torch.log(prior.std() / 10)
        print("prior_log_sigma:", prior_log_sigma)
        self.enc_fc1.reset_parameters(prior, prior_log_sigma)

    def bnn_loss(self):
        return self.enc_fc1.bayesian_kld_loss()

    def freeze(self):
        self.enc_fc1.freeze()

    def unfreeze(self):
        self.enc_fc1.unfreeze()

    def encoder(self, x, knn_graph=None):
        x = torch.cat([self.enc_fc1(x), self.enc_fc2(x)], -1)

        x, enc_attn1 = self.enc_attn1(x, x, x, knn_graph)
        #         x = F.relu(x, True)
        #         x = F.elu(x)

        x, enc_attn2 = self.enc_attn2(x, x, x, knn_graph)
        #         x = F.relu(x, True)
        #         x = F.elu(x)

        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)

        return mu, logvar, {"attn1": enc_attn1, "attn2": enc_attn2}

    def decoder(self, x, knn_graph=None):
        x, dec_attn1 = self.dec_attn1(x, x, x, knn_graph)
        #         x = F.relu(x, True)
        #         x = F.elu(x)

        x, dec_attn2 = self.dec_attn2(x, x, x, knn_graph)
        #         x = F.relu(x, True)
        #         x = F.elu(x)

        x = self.dec_fc(x)

        return x, {"attn1": dec_attn1, "attn2": dec_attn2}

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)

    def model_train(self, sdata, rdata=None, anno_key="anno", epochs=50, lr=1e-3, batchsize=6400, weight_decay=1e-4,
                    r=0.5, alpha=1.0, beta=1.0, gamma=1.0, kappa=1.0, device=torch.device("cuda:0")):
        avoid_overtrain = True if sdata.shape[0] > 15000 else False

        # construct kNN graph with spatial_mtx
        spatial_mtx = sdata.obsm["spatial"].copy()
        knn_graph = kneighbors_graph(spatial_mtx, n_neighbors=self.k_neighbors, include_self=False)
        # spatial spatial prior weighted graph
        metric_graph = spatial_prior_graph(spatial_mtx, self.k_neighbors)
        knn_graph = knn_graph + sp.eye(knn_graph.shape[0])

        dataset = StDataset(sdata.X.copy(), knn_graph, metric_graph)
        index_loader = Ripplewalk_sampler(knn_graph, r=r, batchsize=batchsize, total_times=10)

        # set prior
        if rdata is None:
            sc.tl.pca(sdata, n_comps=self.d_prior)
            prior_weight = torch.FloatTensor(sdata.varm["PCs"].T.copy())
            self.prior_initialize(prior_weight)
        else:
            rdata = integration(rdata, sdata)
            rdata = preprocess(rdata, min_cells=None, target_sum=None, is_filter_MT=True, n_tops=None)
            rdata = get_bulk(rdata, key=anno_key, min_samples=self.d_prior + 1)
            sc.tl.pca(rdata, n_comps=self.d_prior)
            prior_weight = torch.FloatTensor(rdata.varm["PCs"].T.copy())
            self.prior_initialize(prior_weight)

        self.unfreeze()
        self.to(device)
        optimizer = torch.optim.Adam(optim_parameters(self), lr=lr, weight_decay=weight_decay)
        s = time.time()

        last_loss = 1e6

        loss_recorded = 1e6
        overtrain_count = 0
        converge = False

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            count = 0
            for index in index_loader:
                ori, knn_subgraph, neighbor_subgraph = dataset.get_batch(index)
                ori = ori.to(device)
                knn_subgraph = knn_subgraph.to(device)
                neighbor_subgraph = neighbor_subgraph.to(device)

                recons, mu, logvar, enc_attn, dec_attn = self.forward(ori, knn_subgraph)
                amse = loss_amse(recons, ori)
                vae = loss_kld(mu, logvar)
                metric = loss_metric(mu, neighbor_subgraph)
                bnn = self.bnn_loss()
                loss = alpha * vae + beta * amse + gamma * metric + kappa * bnn

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if avoid_overtrain:
                    if loss.item() - loss_recorded > 1e-5:
                        overtrain_count += 1
                    else:
                        overtrain_count = 0

                    if overtrain_count > 5:
                        print("Early Stop")
                        converge = True
                        break

                    loss_recorded = loss.item()

                total_loss += loss.item()
                count += 1

            if abs(total_loss / count - last_loss) < 1e-2 or total_loss / count - last_loss > 1e-2 or converge:
                print("Model Converge")
                break

            last_loss = total_loss / count

            e = time.time()
            print("Epoch:%d Time:%.2fs Loss: %f" % (epoch + 1, e - s, total_loss / count))
            s = time.time()
        self.freeze()

    def output(self, sdata, key_added="embedding", device=torch.device("cpu")):
        knn_graph = kneighbors_graph(sdata.obsm["spatial"], n_neighbors=self.k_neighbors, include_self=False)
        knn_graph = torch.FloatTensor(knn_graph.toarray() + np.eye(knn_graph.shape[0]))

        self.to(device)
        self.eval()
        exp_mtx = torch.FloatTensor(sdata.X.toarray()).to(device)
        knn_graph = knn_graph.to(device)
        with torch.no_grad():
            mu, logvar, enc_attn = self.encoder(exp_mtx, knn_graph)
        enc_attn["attn1"] = sp.csr_matrix(enc_attn["attn1"].cpu().numpy() * knn_graph.cpu().numpy())
        enc_attn["attn2"] = sp.csr_matrix(enc_attn["attn2"].cpu().numpy() * knn_graph.cpu().numpy())

        sdata.obsm[key_added] = mu.cpu().numpy()

        return sdata

    def forward(self, x, knn_graph):
        mu, logvar, enc_attn = self.encoder(x, knn_graph)
        x_lat = self.reparameterize(mu, logvar)
        x_cons, dec_attn = self.decoder(x_lat, knn_graph)

        return x_cons, mu, logvar, enc_attn, dec_attn