import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from diffusion_utilities import ContextUnet

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02
# construct the noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
ab_t[0] = 1

# embedding context (number: 0-9) to a 10-dimensional vector
def embed_fc(x, n_cfeat):
    x = F.one_hot(x, num_classes=10)
    x = x.float()
    x = x.view(-1, 10)
    return x

# define sampling function for DDIM
def denoise_ddim(x, t, t_prev, pred_noise):
    ab = ab_t[t]
    ab_prev = ab_t[t_prev]
    
    x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
    dir_xt = (1 - ab_prev).sqrt() * pred_noise

    return x0_pred + dir_xt

# sample quickly using DDIM
@torch.no_grad()
def sample_ddim(nn_model, n_sample, context, height, timesteps, n=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 1, height, height).to(device)  

    # array to keep track of generated steps for plotting
    intermediate = [] 
    step_size = timesteps // n
    for i in range(timesteps, 0, -step_size):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        eps = nn_model(samples, t, c=context)    # predict noise e_(x_t,t)
        samples = denoise_ddim(samples, i, i - step_size, eps)
        intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

def main():
    n_feat = 64 # 64 hidden dimension feature
    n_cfeat = 10 # context vector is of size 5

    height = 28 # 28x28 image
    save_dir = './weights/'

    # construct model
    nn_model = ContextUnet(in_channels=1, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

    # load in model weights and set to eval mode
    nn_model.load_state_dict(torch.load(f"{save_dir}/model_99.pt", map_location=device))
    nn_model.eval()
    print("Loaded in Model")

    # context label
    original_context = [0,0,0,0,0,0,0,0,0,0]

    # Show numble of 0-9, each with 10 samples
    n_row = 10
    n_col = 10
    img = []
    for i in range(10):
        print(f"Sampling digit {i}")
        context = embed_fc(torch.tensor([i]*10), n_cfeat).to(device)
        samples, intermediate = sample_ddim(nn_model, 10, context, height, timesteps)
        samples = samples.cpu().numpy()
        img.append(samples)
    img = np.array(img)
    img = img.reshape(100, 28, 28)
    fig, axs = plt.subplots(n_row, n_col, figsize=(5, 5))
    cnt = 0
    for i in range(n_row):
        for j in range(n_col):
            axs[i, j].imshow(img[cnt], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.show()


if __name__ == "__main__":
    main()