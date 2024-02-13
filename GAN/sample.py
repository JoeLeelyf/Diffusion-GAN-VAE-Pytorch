import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable

from cgan_utilities import Generator

def main():
    in_channels = 1
    img_size = 28
    n_classes = 10
    latent_dim = 100

    # Initialize generator and discriminator
    generator = Generator(in_channels, n_classes, latent_dim, img_size)
    generator.load_state_dict(torch.load('weights/generator_100.pt'))
    generator.eval()

    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    # Show numble of 0-9, each with 10 samples
    n_row = 10
    n_col = 10
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row * n_col, latent_dim))))
    labels = np.array([num for _ in range(n_col) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)

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