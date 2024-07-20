import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate):
        super(Encoder, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
        self.batch_norm2 = nn.BatchNorm1d(latent_dim)
        self.act2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = self.batch_norm2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate):
        super(Decoder, self).__init__()
        
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.linear3 = nn.Linear(hidden_dim, input_dim)
        self.act3 = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = self.batch_norm2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        
        x = self.linear3(x)
        x = self.act3(x)
        return x

class HVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate):
        super(HVAE, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout_rate)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, dropout_rate)
        
    def reparameterization(self, mean, logvar):
        # vMF distribution sampling
        u = torch.randn(mean.shape).to(mean.device)
        u = u / torch.norm(u, dim=-1, keepdim=True)  # normalize to lie on the unit sphere
        z = mean + u * torch.exp(0.5 * logvar)
        return z
    
    def forward(self, x):
        #x = x.reshape(-1, 1, 28, 28)
        mean, logvar = self.encoder(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar

