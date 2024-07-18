import torch
from torch import nn
import torch.nn.functional as F
from model.Encoder import AutoEncoder
from model.Decoder import AutoDecoder

class Encoder1(nn.Module):
    def __init__(
        self, 
        latent_dim,
        input_dim,
        hidden_dim,
        dropout_rate
        ):
        
        super(Encoder1, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
        self.batch_norm2 = nn.BatchNorm1d(latent_dim)
        self.act2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

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

class Decoder1(nn.Module):
    def __init__(
        self, 
        latent_dim,
        input_dim,
        hidden_dim,
        dropout_rate
        ):
        
        super(Decoder1, self).__init__()
        
        self.linear1 = nn.Linear(2, latent_dim)
        self.batch_norm1 = nn.BatchNorm1d(latent_dim)
        self.act1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.linear2 = nn.Linear(latent_dim, hidden_dim)
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
        
class VAE1(nn.Module):
    
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        latent_dim,
        dropout_rate
        ):
        
        super(VAE1, self).__init__()

        # encoder
        self.encoder = Encoder1(
                            input_dim=input_dim, 
                            hidden_dim=hidden_dim, 
                            latent_dim=latent_dim, 
                            dropout_rate=dropout_rate
                            )
        
        self.decoder = Decoder1(
                            input_dim=input_dim, 
                            hidden_dim=hidden_dim, 
                            latent_dim=latent_dim, 
                            dropout_rate=dropout_rate
                            )
        
    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar

    

class VAE(nn.Module):

    def __init__(
        self, 
        input_dim=784, 
        hidden_dim=400, 
        latent_dim=200, 
        device='cpu'):
        
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)#.to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = AutoEncoder(latent_dims)
        self.decoder = AutoDecoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

