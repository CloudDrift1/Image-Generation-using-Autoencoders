import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims = [16,32,64],expand_dim=4):
        """
        Initialize the decoder model.

        Args:
            latent_dim : dimension of latent vector (type : int)
            hidden_dims : list of hidden layer dimensions (type : list, default : None)
                - reverse order of encoder hidden layer dimensions
            expand_dim : size of the first hidden layer input of self.decoder (type : int)
                - the first hidden layer input of self.decoder is (B, self.hidden_dims[-1], self.expand_dim, self.expand_dim)
        """
        super(Decoder, self).__init__()
        self.hidden_dims = hidden_dims
        self.expand_dim = expand_dim
        self.latent_dim = latent_dim
        self.input_layer = nn.Linear(self.latent_dim, self.hidden_dims[-1] * (self.expand_dim ** 2))

        self.decoder = nn.Sequential(nn.ConvTranspose2d(self.hidden_dims[3], self.hidden_dims[2], kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(self.hidden_dims[2], self.hidden_dims[1], kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(self.hidden_dims[1], self.hidden_dims[0], kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())

        self.last_layer = nn.Sequential(nn.ConvTranspose2d(self.hidden_dims[0], 1, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid())

    def forward(self, x):
        """
        Foward pass of the decoder.

        Args:
            x : the input to the decoder (latent vector) (type : torch.Tensor, size : (batch_size, latent_dim))
        Returns:
            out : the output of the decoder (type : torch.Tensor, size : (batch_size, 1, 16, 16))
        """

        x = self.input_layer(x)
        x = x.view(-1, self.hidden_dims[-1], self.expand_dim, self.expand_dim)
        x = self.decoder(x)
        out = self.last_layer(x)

        return out