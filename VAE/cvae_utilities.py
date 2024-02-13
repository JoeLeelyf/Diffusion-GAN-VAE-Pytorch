import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple

class CVAE(nn.Module):
    def __init__(self, 
                 channels: int, 
                 n_classes: int,
                 latent_dim: int, 
                 img_size: int
                 ):
        super(CVAE, self).__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.n_classes = n_classes

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(channels * img_size * img_size + n_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(512, latent_dim)
        self.fc3 = nn.Linear(512, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, channels * img_size * img_size),
            nn.Sigmoid(),
        )
        self.fc5 = nn.Linear(512, channels * img_size * img_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h1 = self.encoder(inputs)

        z_mu = self.fc2(h1)
        z_var = self.fc3(h1)
        
        return z_mu, z_var
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)

        return self.decoder(inputs).view(-1, self.channels, self.img_size, self.img_size)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, self.channels * self.img_size * self.img_size), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
