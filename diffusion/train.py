from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

import os

from diffusion_utilities import UNet, DiffusionUtil

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def save_images(images, path, n_row=10):
    grid = make_grid(images, nrow=n_row)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def train():
    os.makedirs("images", exist_ok=True)

    # parameters
    timesteps = 500
    beta1 = 1e-4
    beta2 = 0.02

    # network hyperparameters
    n_classes = 10
    img_size = 32

    # training hyperparameters
    batch_size = 200
    n_epoch = 100
    lrate=1e-3

    # construct model
    nn_model = UNet(in_channels=1, out_channels=1, img_size=img_size, \
                    num_classes=n_classes, device=device).to(device)
    diffusion_util = DiffusionUtil(timesteps, beta1, beta2, img_size=img_size, device=device)

    # optimizer
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    # Load MINIST dataset
    dataset = MNIST(
        root='../data/', 
        download=True, 
        transform=transforms.Compose(
            # Notice that normalization is very important here
            [
                # Reshape to 32*32
                transforms.Resize((img_size, img_size)),
                ToTensor(), 
                transforms.Normalize([0.5], [0.5]),
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    nn_model.train()
    losses = []
    save_dir = './weights/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Training starts")
    for epoch in range(n_epoch):
        loss_ep = 0
        for imgs, labels in tqdm(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            t = diffusion_util.sample_timesteps(batch_size).to(device)
            x_t, noise = diffusion_util.noise_images(imgs, t)

            # Randomly drop labels
            if np.random.rand() < 0.1:
                labels = None

            # Forward
            pred = nn_model(x_t, t, labels)
            loss = F.mse_loss(pred, noise)

            # Backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_ep += loss.item()
        
        avg_loss = loss_ep / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch} Loss: {avg_loss}")

        # Save the model
        if epoch % 10 == 0:
            n_row = 10
            labels = np.array([num for _ in range(n_row) for num in range(n_row)])
            labels = torch.tensor(labels).to(device)
            sample = diffusion_util.sample(nn_model, 1, n_row * n_row, labels)
            save_images(sample, f"images/{epoch}.png", n_row=n_row)
            torch.save(nn_model.state_dict(), save_dir + f'epoch_{epoch}.pth')
            print(f"Model saved at epoch {epoch}")
        
    return losses
            

def main():
    losses = train()
    # Draw the loss curve
    plt.plot(losses)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Diffusion_loss.png')

if __name__ == "__main__":
    main()