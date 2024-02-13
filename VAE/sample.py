import torch
import numpy as np
import matplotlib.pyplot as plt

from cvae_utilities import CVAE
from train import one_hot

def main():
    channels = 1
    img_size = 28
    n_classes = 10
    latent_dim = 20

    # Initialize generator and discriminator
    cvae = CVAE(channels, n_classes, latent_dim, img_size)
    cvae.load_state_dict(torch.load('weights/cvae_50.pt'))
    cvae.eval()

    # Show numble of 0-9, each with 10 samples
    n_row = 10
    n_col = 10
    z = torch.randn(n_row * n_col, latent_dim)
    labels = np.array([num for _ in range(n_col) for num in range(n_row)])
    labels = one_hot(torch.tensor(labels), n_classes)

    gen_imgs = cvae.decode(z, labels).cpu().detach()

    # Plot the generated images
    fig, axs = plt.subplots(n_row, n_col, figsize=(5, 5))
    cnt = 0
    for i in range(n_row):
        for j in range(n_col):
            axs[i, j].imshow(gen_imgs[cnt].cpu().detach().numpy().reshape(28, 28), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.show()

if __name__ == "__main__":
    main()