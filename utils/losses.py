import torch
import numpy as np

def reconstruction_loss(recon_x, x):
    """
    Returns the reconstruction loss of VAE.
    
    Args:
        recon_x : reconstructed x (type : torch.Tensor, size : (batch_size, 1, 16, 16)
        x : original x (type : torch.Tensor, size : (batch_size, 1, 16, 16)
    Returns:
        recon_loss : reconstruction loss (type : torch.Tensor, size : (1,))
    """
    loss = 0.0
    eps = 1e-18
    batch_size = x.size(0)

    #negative log-likelihood for each pixel
    bce_loss = - (x * torch.log(recon_x + eps) + (1 - x) * torch.log(1 - recon_x + eps)) #without using torch.nn
    loss = torch.sum(bce_loss)
    loss = loss / batch_size

    return loss

def KLD_loss(mu, logvar):
    """
    Returns the regularization loss of VAE.
    
    Args:
        mu : latent mean (type : torch.Tensor, size : (batch_size, latent_dim))
        logvar : latent log variance (type : torch.Tensor, size : (batch_size, latent_dim))
    Returns:
        kld_loss : regularization loss (type : torch.Tensor, size : (1,))
    """

    batch_size = mu.size(0)

    #KL Divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld_loss = torch.sum(kld_loss) 
    kld_loss = kld_loss / batch_size

    return kld_loss

def loss_function(recon_x, x, mu, logvar,beta=1,return_info=False):
    """
    Returns the loss of beta-VAE.

    Args:
        recon_x : reconstructed x (type : torch.Tensor, size : (batch_size, 1, 16, 16)
        x : original x (type : torch.Tensor, size : (batch_size, 1, 16, 16)
        mu : latent mean (type : torch.Tensor, size : (batch_size, latent_dim))
        logvar : latent log variance (type : torch.Tensor, size : (batch_size, latent_dim))
        beta : beta value for beta-VAE (type : float)
    Returns:
        loss : loss of beta-VAE (type : torch.Tensor, size : (1,))
            - Reconstruction loss + beta * Regularization loss
            Recon_loss : reconstruction loss
               kld_loss : KL divergence loss
    """    

    Recon_loss = reconstruction_loss(recon_x, x)
    kld_loss = KLD_loss(mu, logvar)
    loss = Recon_loss + beta * kld_loss

    if return_info:
        return {"loss" : loss,
                "recon_loss" : Recon_loss,
                "kld_loss" : kld_loss}
    else :
        return loss
