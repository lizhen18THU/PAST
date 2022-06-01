import torch.nn as nn
import torch

def loss_amse(recons, ori):
    SE_keepdim = nn.MSELoss(reduction="none")

    Gamma = ori.data.sign().absolute()
    Q = Gamma.mean(dim=1)
    Gamma = Gamma + (Gamma - 1).absolute() * Q.reshape(-1, 1)

    loss = SE_keepdim(ori, recons) * Gamma

    return loss.mean()

def loss_kld(mu, logvar):
    """
    mu: latent mean
    logvar: latent log variance
    """
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5) / mu.shape[0]

    return KLD

def loss_metric(latent, attn):
    dist = torch.cdist(latent, latent, p=2) ** 2
    loss = torch.mean(torch.sum(attn * dist, dim=-1))

    return loss