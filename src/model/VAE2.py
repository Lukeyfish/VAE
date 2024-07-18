import torch
from torch import nn
import torch.nn.functional as F

class Reshape(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.view(self.dim)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList()
        for hdim in hidden_dims:
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, hdim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hdim),
                    nn.LeakyReLU(1e-2)
                )
            )
            input_dim = hdim
        
        self.flatten = nn.Flatten()
        self.mean_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims):
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        self.layers.append(Reshape((-1, hidden_dims[-1], 1, 1)))
        
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i+1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU(1e-2)
                )
            )
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=4, padding=4),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(1e-2),
            nn.Conv2d(hidden_dims[-1], out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)

class VAE2(nn.Module):
    def __init__(self, latent_dim, input_dim=1, hidden_dims=[32, 64, 128, 256, 512], device='cuda'):
        super(VAE2, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims)
        self.device = device

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar
