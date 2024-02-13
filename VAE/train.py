import os
import numpy as np

from torchvision.utils import save_image

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import torch

import matplotlib.pyplot as plt

from tqdm import tqdm

from cvae_utilities import CVAE

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train():
    os.makedirs("images", exist_ok=True)

    # Parameters
    channels = 1
    latent_dim = 20
    img_size = 28
    n_epochs = 50
    batch_size = 64
    n_classes = 10

    # Load MINIST dataset
    dataset = MNIST(
        root='../data/', 
        download=True, 
        transform=ToTensor(),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize CVAE model
    model = CVAE(channels, n_classes, latent_dim, img_size).to(device)

    # Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Training starts")
    save_dir = "weights/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    losses = []
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            
            labels = one_hot(labels, n_classes)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            reconstructed_images, mu, logvar = model(images, labels)
            
            # Compute total loss
            loss = loss_function(reconstructed_images, images, mu, logvar)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {average_loss:.4f}")
        losses.append(average_loss)

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{save_dir}cvae_{epoch}.pt")
            print(f"Model saved at epoch {epoch}")
            # Generate and save a sample
            with torch.no_grad():
                n_row = 10
                z = torch.randn(n_row ** 2, latent_dim).to(device)
                labels = np.array([num for _ in range(n_row) for num in range(n_row)])
                labels = one_hot(torch.tensor(labels), n_classes).to(device)
                generated_image = model.decode(z, labels).cpu().detach()

                save_image(generated_image, f"images/{epoch}.png", nrow=n_row, normalize=True)
    
    print("Training finished")
    return losses

def main():
    losses = train()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("VAE_loss.png")

if __name__ == "__main__":
    main() 
