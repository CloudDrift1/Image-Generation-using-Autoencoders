import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,in_channels=1, hidden_dims = [16,32,64], latent_dim=2,
              model_name='VAE'):
        """
        Initialize the encoder model.

        Args:
            in_channels : number of channels of input image
            hidden_dims : list of hidden layer dimensions
            latent_dim : dimension of latent vector
            model_name : type of model (beta-VAE or AE)
        """
        super(Encoder, self).__init__()
        
        self.hidden_dims = hidden_dims
        self.model_name = model_name
        self.in_channels = in_channels

        self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels=self.hidden_dims[0], kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(self.hidden_dims[0], out_channels=self.hidden_dims[1], kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(self.hidden_dims[1], out_channels=self.hidden_dims[2], kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(self.hidden_dims[2], out_channels=self.hidden_dims[3], kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())

       
        self.fc_mean = nn.Linear(self.hidden_dims[3], latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dims[3], latent_dim)
        

    def reparametrize(self, mu, logvar, eps):
        """
        Returns the reparametrized latent vector.

        Args:
            mu : latent mean (type : torch.Tensor, size : (batch_size, latent_dim))
            logvar : latent log variance (type : torch.Tensor, size : (batch_size, latent_dim))
            eps : random noise for encoder (type : torch.Tensor, size : (batch_size, latent_dim))
        Returns:
            rp : reparametrized latent vector (type : torch.Tensor, size : (batch_size, latent_dim))
        """

        std = torch.exp(0.5 * logvar)
        rp = mu + eps * std
        return rp
    
    def sample_noise(self, logvar):
        return torch.randn_like(logvar)
    
    def forward(self, x, eps=None):
        """
        Forward pass of the encoder.

        Args:
            x : the input to the encoder (image) (type : torch.Tensor, size : (batch_size, 1, 16, 16))
            eps : random noise for encoder (type : torch.Tensor, size : (batch_size, latent_dim))
        Returns:
            For VAE, return mu, logvar, rp
                mu : latent mean (type : torch.Tensor, size : (batch_size, latent_dim))
                logvar : latent log variance (type : torch.Tensor, size : (batch_size, latent_dim))
                rp : reparametrized latent vector (type : torch.Tensor, size : (batch_size, latent_dim))
            For AE, return out
                out : latent vector (type : torch.Tensor, size : (batch_size, latent_dim))
        """

        x = self.model(x)
        x = x.view(x.size(0), -1)

        if self.model_name == 'VAE':
            mu = self.fc_mean(x)
            logvar = self.fc_logvar(x)
            eps = eps if eps is not None else self.sample_noise(logvar)
            rp = self.reparametrize(mu, logvar, eps)
            return mu, logvar, rp

        elif self.model_name == 'AE':
            out = self.fc_mean(x)
            return out