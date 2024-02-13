import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.autograd import Variable

import matplotlib.pyplot as plt
from tqdm import tqdm

from cgan_utilities import Generator, Discriminator

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train():
    os.makedirs("images", exist_ok=True)

    # Parameters
    in_channels = 1
    img_size = 28
    n_classes = 10
    latent_dim = 100
    n_epochs = 100
    batch_size = 64

    # Load MINIST dataset
    dataset = MNIST(
        root='../data/', 
        download=True, 
        transform=transforms.Compose(
            # Notice that normalization is very important here
            [ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss function
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = Generator(in_channels, n_classes, latent_dim, img_size).to(device)
    discriminator = Discriminator(in_channels, n_classes, img_size).to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    G_losses = []
    D_losses = []

    def sample_image(n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = Variable(LongTensor(labels))
        gen_imgs = generator(z, labels)
        save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

    print("Training starts")
    save_dir = 'weights/'
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        for imgs, labels in tqdm(dataloader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor)).to(device)
            labels = Variable(labels.type(LongTensor)).to(device)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

        print(f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            sample_image(n_row=10, batches_done=epoch)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(generator.state_dict(), f'{save_dir}/generator_{epoch}.pt')
            torch.save(discriminator.state_dict(), f'{save_dir}/discriminator_{epoch}.pt')
            print(f"Model saved at {save_dir}/generator_{epoch}.pt")
            print(f"Model saved at {save_dir}/discriminator_{epoch}.pt")
    
    print("Training finished")
    return G_losses, D_losses

def main():
    G_losses, D_losses = train()
    # Plot the loss
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("GAN_loss.png")

if __name__ == "__main__":
    main()