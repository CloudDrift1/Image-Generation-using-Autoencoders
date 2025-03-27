# MNIST Digit Generation using Autoencoder (AE), Variational Autoencoder (VAE) and Beta-VAE

This repository contains the implementation of a Variational Autoencoder (VAE), Beta-VAE, and Autoencoder (AE) used for generating MNIST digits. The code uses PyTorch and is designed to train these models, visualize the generated images, and save the results as a GIF. 

## Models

- **Variational Autoencoder (VAE)**: A probabilistic model designed to generate data by learning the distribution of the data in a latent space.
- **Beta-VAE**: A variant of the VAE where the KL-divergence term is scaled by a hyperparameter `beta` to enforce better disentangled representations.
- **Autoencoder (AE)**: A simpler version of VAE, where we do not use the probabilistic approach and just train the model to reconstruct the input.

## Requirements

To install the required libraries, use:

```bash
pip install -r requirements.txt
```

## Model Overview
### Encoder
The encoder uses convolutional layers to map the input image into a lower-dimensional latent space. For VAE and Beta-VAE, it outputs the mean and log variance for the latent distribution. For AE, it outputs a deterministic latent vector.

### Decoder
The decoder takes the latent vector and applies transpose convolutional layers to reconstruct the image from the latent space.

## Training
* Loss function: For VAE and Beta-VAE, the loss consists of a reconstruction loss (MSE) and a KL-divergence term. For AE, only the reconstruction loss is used.

* Optimizer: Adam optimizer is used for training both models.

* Beta-VAE: The beta parameter controls the weight of the KL-divergence term in the loss function.

### Training Procedure
The training loop involves:

Forward pass through the encoder and decoder.

Backpropagation of the total loss (reconstruction loss + KL loss for VAE and Beta-VAE).

Visualization of generated images during training.

Usage
Training VAE or Beta-VAE or AE
You can train any of the models by modifying the model_name parameter when initializing the train_VAE class. Here is an example of training a Beta-VAE model:

```python
from model import Encoder, Decoder, train_VAE

encoder = Encoder(in_channels=1, hidden_dims=[16, 32, 64], latent_dim=2, model_name='beta_VAE')
decoder = Decoder(latent_dim=2, hidden_dims=[64, 32, 16])

trainer = train_VAE(train_loader, test_loader, encoder, decoder, device, config, model_name='beta_VAE', beta=4)
trainer.train()
```

The loss curve looks as follows,

![image](https://github.com/user-attachments/assets/bb601a8d-ede9-4645-b42c-61476aba3e3a)

Configuration
The config argument is expected to be an object with the following fields:

* epoch: Number of epochs for training.

* lr: Learning rate for the optimizer.

* latent_dim: Dimensionality of the latent space.

* batch_size: Batch size for training.

## Generating Images
After training, you can generate new images by calling get_fake_images:

```python
fake_images = trainer.get_fake_images(test_images, test_labels, eps)
```

### Saving Generated Images as GIF
To save the generated images as a GIF during training, set save_img=True when initializing the train_VAE class. The GIF will be saved in the working directory.

```python
trainer = train_VAE(train_loader, test_loader, encoder, decoder, device, config, save_img=True)
trainer.train()
```


## Results

Sample GIF,

![VAE_generated_img](https://github.com/user-attachments/assets/94a6fd05-0b20-4328-9c7c-9c8efdf8eb2d)

The trained model can generate MNIST digit images, and the generated images are saved as GIFs during training (if enabled). Below is an example of the generated images from the Beta-VAE model:

![image](https://github.com/user-attachments/assets/e1e5afe4-f843-4532-8107-b3daf2e1cd26)

Cpmparison of AE, VAE and beta VAE generated images,

![image](https://github.com/user-attachments/assets/5057b95c-9fa2-41ef-891e-170cac5ce0fd)

Latent vector comparison

![image](https://github.com/user-attachments/assets/3338accc-53f9-438e-9fe0-b99c647f0abd)

Image quality relation to latent vector dimension,

![image](https://github.com/user-attachments/assets/406ca882-cc74-4180-bf56-b039c8f5e7e0)


### Notes
1. The models were trained on the MNIST dataset. Ensure that you have the MNIST dataset available or use a DataLoader to load it.

2. For better disentanglement of the latent space, consider experimenting with the beta parameter in Beta-VAE.

3. Visualization of the generated images during training can be controlled using the img_show flag.

