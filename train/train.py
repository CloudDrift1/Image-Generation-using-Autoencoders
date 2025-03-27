import torch
import torch.nn as nn
import numpy as np
import torchvision
from utils.utils import *
from utils.losses import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

class train_VAE:
    def __init__(self, train_loader, test_loader, encoder, decoder, device,
                 config, save_img = False, model_name='VAE',beta=1, img_show=False):
        """"
        Initialize the training_VAE class.

        Args:
            train_loader : the dataloader for training dataset (type : torch.utils.data.DataLoader)
            test_loader : the dataloader for test dataset (type : torch.utils.data.DataLoader)
            encoder : the encoder model (type : Encoder)
            decoder : the decoder model (type : Decoder)
            device : the device where the model will be trained (type : torch.device)
            config : the configuration for training (type : SimpleNamespace)
            save_img : whether to save the generated images during training
            model_name : type of model - VAE or AE
                - VAE includes VAE and beta-VAE
            beta : beta value for beta-VAE (type : float)
            img_show : whether to show the generated images during training
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = config.epoch
        self.lr = config.lr
        self.latent_dim = config.latent_dim
        self.batch_size = config.batch_size
        self.device = device

        self.generated_img = []
        self.Recon_loss_history = []
        self.KLD_loss_history = []
        self.system_info = getSystemInfo()
        self.save_img = save_img
        self.model_name = model_name

        if self.model_name == 'beta_VAE':
            self.model_name = f"{self.model_name}_{beta}"
        self.beta = beta
        self.img_show = img_show
        self.encoder = encoder
        self.decoder = decoder

        self.optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=self.lr
        )
 
    def make_gif(self):
        """
        Save the generated images as a gif file.
        """
        if len(self.generated_img) <= 1:
            print("No frame to save")
            return
        else :
            print("Saving gif file...")
            for i in range(len(self.generated_img)):
                self.generated_img[i] = Image.fromarray(self.generated_img[i])
            self.generated_img[0].save(f"./{self.model_name}_generated_img.gif",
                                save_all=True, append_images=self.generated_img[1:], 
                                optimize=False, duration=700, loop=1) 
    def one_iter_train(self, images,label,eps):
        """
        Train the model for one iteration.

        Args:
            
            images : the input images (type : torch.Tensor, size : (batch_size, 1, 16, 16))
            label : the input labels (type : torch.Tensor, size : (batch_size))
            eps : random noise for encoder (type : torch.Tensor, size : (batch_size, latent_dim))
                - it is used in reparametrization trick
        Returns:
            dict : dictionary of losses
                - VAE :
                recon_loss : the reconstruction loss of the model (type : torch.Tensor, size : (1,))
                kld_loss : the regularization (KL divergence) loss of the model (type : torch.Tensor, size : (1,))
                - AE :
                recon_loss : the reconstruction loss of the model (type : torch.Tensor, size : (1,))
        """

        if self.model_name in ['VAE', 'beta_VAE']:
          mu, logvar, rp = self.encoder(images, eps)
        else:
          rp = self.encoder(images, eps)

        recon_x = self.decoder(rp)
        if self.model_name in ['VAE', 'beta_VAE']:
          loss_info = loss_function(recon_x, images, mu, logvar, beta=self.beta, return_info=True)
          recon_loss = loss_info["recon_loss"]
          kld_loss = loss_info["kld_loss"]
          loss = loss_info["loss"]
        else:
          loss = reconstruction_loss(recon_x, images)
          recon_loss = loss
          kld_loss = torch.tensor(0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
                "recon_loss" : recon_loss.item(),
                "kld_loss" : kld_loss.item()
                }
    
    def get_fake_images(self, image,labels,eps):
        self.encoder.eval()
        self.decoder.eval()
        if self.model_name == 'AE':
            with torch.no_grad():
                rp = self.encoder(image)
                fake_images = self.decoder(rp)
        elif 'VAE' in self.model_name:
            with torch.no_grad():
                mean, logvar, rp = self.encoder(image,eps)
                fake_images = self.decoder(rp)
        else:
            raise NotImplementedError(f"Please choose the model type in ['VAE', 'AE'], not {self.model_name}")
        return fake_images
    
    def train(self):
        """
        Train the VAE model.
        """
        try : 
            for epoch in range(1,self.num_epochs+1):
                pbar = tqdm(enumerate(self.train_loader,start=1), total=len(self.train_loader))
                for i, (images, labels) in pbar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    eps = torch.randn(self.batch_size, self.latent_dim).to(self.device)
                    if epoch == 1 and i == 1:
                        fake_images = self.get_fake_images(images,labels,eps)
                        grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True).detach().cpu().permute(1,2,0).numpy()
                        self.generated_img.append((grid_img* 255).astype('uint8'))
                    self.encoder.train()
                    self.decoder.train()
                    results = self.one_iter_train(images,labels,eps)
                    self.encoder.eval()
                    self.decoder.eval()
                    recon_loss, kld_loss = results['recon_loss'], results['kld_loss']
                    self.Recon_loss_history.append(recon_loss)
                    self.KLD_loss_history.append(kld_loss)
                    
                    fake_images = self.get_fake_images(images, labels,eps)

                    pbar.set_description(
                        f"Epoch [{epoch}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], Total loss : {recon_loss + kld_loss:.6f} Recon Loss: {recon_loss:.6f}, KLD Loss: {kld_loss:.6f}")
                
                grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True).detach().cpu().permute(1,2,0).numpy()
                self.generated_img.append((grid_img* 255).astype('uint8'))
                if self.img_show:
                    plt.imshow(grid_img)
                    plt.pause(0.01)
  
        except KeyboardInterrupt:
            print('Keyboard Interrupted, finishing training...')
        if self.save_img:
            self.make_gif() 
        
        return {'encoder' : self.encoder,
                'encoder_state_dict' : self.encoder.state_dict(),
                'decoder' : self.decoder,
                'decoder_state_dict' : self.decoder.state_dict(),
                'Recon_loss_history' : self.Recon_loss_history,
                'generated_img' : self.generated_img[-1],
                'KLD_loss_history' : self.KLD_loss_history}