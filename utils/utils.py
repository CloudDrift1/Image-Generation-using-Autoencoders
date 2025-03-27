import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import imageio.v2 as imageio
import platform,socket,re,uuid,json,psutil,logging
import random
from turtle import width
from types import SimpleNamespace


def get_MNIST_image_and_visualize():
    image1 = torch.Tensor(cv2.imread('./test_file/GAN_0000.png',cv2.IMREAD_GRAYSCALE)).unsqueeze(0).unsqueeze(0)
    image2 = torch.Tensor(cv2.imread('./test_file/GAN_0050.png',cv2.IMREAD_GRAYSCALE)).unsqueeze(0).unsqueeze(0)
    image3 = torch.Tensor(cv2.imread('./test_file/GAN_0100.png',cv2.IMREAD_GRAYSCALE)).unsqueeze(0).unsqueeze(0)
    plt.subplot(131)
    plt.title("EPOCH 000")
    plt.imshow(image1.squeeze())
    plt.subplot(132)
    plt.title("EPOCH 050")
    plt.imshow(image2.squeeze())
    plt.subplot(133)
    plt.title("EPOCH 100")
    plt.imshow(image3.squeeze())
    plt.pause(0.01)
    
    return image1, image2, image3

def set_randomness(seed : int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_weights(model):
    classname = model.__class__.__name__
    # fc layer
    if classname.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    # conv layer
    elif classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        if model.bias is not None:
            nn.init.constant_(model.bias.data, 0)
    # batchnorm
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def show_image_with_label(gen,config=None,device='cuda',cols=4,rows=4):
    if config is None:
        config = SimpleNamespace(
                latent_dim = 100
        )
    with torch.no_grad():
        fixed_noise = torch.randn(cols*rows, config.latent_dim).to(device)
        label = torch.zeros((cols*rows,5)).to(device)
        for i in range(cols*rows):
            label[i][i%5] = 1
        img_fake = gen(fixed_noise,label).detach().cpu()
        size_of_figure = (int(cols*1.5),int(rows*1.5))
        fig = plt.figure(figsize=size_of_figure)
        for i in range(rows * cols):
            
            fig.add_subplot(rows, cols, i+1)
            plt.title(torch.argmax(label[i]).item())
            plt.axis('off')
            img_fake[i] = (img_fake[i] - img_fake[i].min())/ (img_fake[i].max() - img_fake[i].min())
            plt.imshow(img_fake[i].permute(1,2,0), cmap='gray')
    plt.show()

def show_image_VAE(model_name : str,decoder,latent_dims, size=10):
    fig = plt.figure(figsize=(5,5))
    fig.suptitle(f"{model_name}")
    for j in range(size):
        for i in range(size):
            z= [[(i-size/2)*5/size,(j-size/2)*5/size]]
            z = torch.FloatTensor(z).cuda()
            z = z.repeat(1,latent_dims//2)
            if latent_dims%2 == 1:
                tmp = torch.tensor([(5+10*i)/size-5]).view(1,1).cuda()
                z = torch.cat((z,tmp),dim=1)
            output = decoder(z)
            ax = fig.add_subplot(size, size, i+1+size*j)
            ax.imshow(torch.squeeze(output.cpu()).data.numpy(), cmap="gray")
            ax.axis("off")

def show_image_real_and_VAE(image,output,num_img=5):
    recon_len = len(output.keys())
    for i in range(num_img):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1+recon_len, 1)
        ax.set_title("Original")
        ax.imshow(torch.squeeze(image[i].cpu()).data.numpy())
        ax.axis("off")
        for j, (key,out_img) in enumerate(output.items()):
            ax = fig.add_subplot(1, 1+recon_len, j+2)
            ax.set_title(f"Reconstructed\nby {key}")
            ax.imshow(out_img[i][0])
            ax.axis("off")

def visualize_latent_dim(trained_encoder,title,sample_size=5000,position='left'):
    
    from HW4_1_YourAnswer import dataloader
    test_loader = dataloader(train=False, batch_size=sample_size)
    for batch_idx, (image, label) in enumerate(test_loader):
        image = (image).cuda()
        if 'VAE' in title:
            mu, log_var, reparam = trained_encoder(image)
        else:
            reparam = trained_encoder(image)
        break
    reparam = reparam.cpu().detach().numpy()
    if position == 'left':
        plt.subplot(121)
    elif position == 'right' :
        plt.subplot(122)
    else :
        pass
    if reparam.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reparam = pca.fit_transform(reparam)
    plt.title(title)
    plt.scatter(reparam[:, 0], reparam[:, 1], c=label, cmap='tab10')
    width = 5 #40
    plt.xlim(-width, width)
    plt.ylim(-width, width)
    plt.colorbar()

def getSystemInfo():
    try:
        info={}
        info['platform']=platform.system()
        info['platform-release']=platform.release()
        info['platform-version']=platform.version()
        info['architecture']=platform.machine()
        info['hostname']=socket.gethostname()
        info['ip-address']=socket.gethostbyname(socket.gethostname())
        info['mac-address']=':'.join(re.findall('..', '%012x' % uuid.getnode()))
        info['processor']=platform.processor()
        info['ram']=str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
        return json.dumps(info)
    except Exception as e:
        logging.exception(e)


def preprocess_image(im):
    original_im = im.clone()
    transform = transforms.Compose([
        transforms.Resize(299),
    ])
    if im.dtype == torch.uint8:
        im = im.astype(torch.float16)  / 255
    elif im.max() > 1.0 or im.min() < 0.0:
        im = (im - im.min()) / (im.max() - im.min())
    im = im.type(torch.float16)
    im = transform(im)
    if im.shape[0] == 1:
        im = im.repeat(3,1,1)
    # print(im.shape)
    try:
        assert im.max() <= 1.0, im.max()
        assert im.min() >= 0.0, im.min()
        assert im.dtype == torch.float16
        assert im.shape == (3, 299, 299), im.shape
    except:
        print("original_im", original_im.shape)
        print(original_im.max(), original_im.min())
        print("transformed image",im.shape)
        print(im.max(), im.min())
        raise AssertionError 
        
    return im
    

def preprocess_images(images):
    """Resizes and shifts the dynamic range of image to 0-1
    Args:
        images: torch.Tensor, shape: (N, 3, H, W), dtype: float16 between 0-1 or np.uint8

    Return:
        final_images: torch.tensor, shape: (N, 3, 299, 299), dtype: torch.float16 between 0-1
    """
    final_images = torch.stack([preprocess_image(im) for im in images], dim=0)
    assert final_images.shape == (images.shape[0], 3, 299, 299)
    assert final_images.max() <= 1.0
    assert final_images.min() >= 0.0
    assert final_images.dtype == torch.float16
    
    return final_images
