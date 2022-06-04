import torch.nn as nn
import torch

def loss_amse(recons, ori):
    """
    Adapted mean square error

    Parameters
    ------
    recons
        reconstructed gene expression matrix
    ori
        original gene expression matrix

    Returns
    ------
    loss
        adapted mean square loss
    """

    SE_keepdim = nn.MSELoss(reduction="none")

    Gamma = ori.data.sign().absolute()
    Q = Gamma.mean(dim=1)
    Gamma = Gamma + (Gamma - 1).absolute() * Q.reshape(-1, 1)

    loss = SE_keepdim(ori, recons) * Gamma

    return loss.mean()

def loss_kld(mu, logvar):
    """
    KL divergence of normal distribution N(mu, exp(logvar)) and N(0, 1)

    Parameters
    ------
    mu
        mean vector of normal distribution
    logvar
        Logarithmic variance vector of normal distribution

    Returns
    ------
    KLD
        KL divergence loss
    """

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5) / mu.shape[0]

    return KLD

def loss_metric(latent, attn):
    """
    Metric learning loss

    Parameters
    ------
    latent
        latent features
    attn
        spatial prior graph

    Returns
    ------
    loss
        metric learning loss
    """

    dist = torch.cdist(latent, latent, p=2) ** 2
    loss = torch.mean(torch.sum(attn * dist, dim=-1))

    return loss