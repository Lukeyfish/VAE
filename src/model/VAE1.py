import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(
        self, 
        latent_dim,
        input_dim,
        hidden_dims,
        dropout_rate
        ):
        
        super(Encoder, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dims[0])
        self.batch_norm1 = nn.BatchNorm1d(hidden_dims[0])
        self.act1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.batch_norm2 = nn.BatchNorm1d(hidden_dims[1])
        self.act2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.mean_layer = nn.Linear(hidden_dims[1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[1], latent_dim)

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
    def __init__(
        self, 
        latent_dim,
        input_dim,
        hidden_dims,
        dropout_rate
        ):
        
        super(Decoder, self).__init__()
        
        self.linear1 = nn.Linear(latent_dim, hidden_dims[1])
        self.batch_norm1 = nn.BatchNorm1d(hidden_dims[1])
        self.act1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.linear2 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.batch_norm2 = nn.BatchNorm1d(hidden_dims[0])
        self.act2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.linear3 = nn.Linear(hidden_dims[0], input_dim)
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
        hidden_dims, 
        latent_dim,
        dropout_rate
        ):
        
        super(VAE1, self).__init__()

        # encoder
        self.encoder = Encoder(
                            input_dim=input_dim, 
                            hidden_dims=hidden_dims, 
                            latent_dim=latent_dim, 
                            dropout_rate=dropout_rate
                            )
        
        self.decoder = Decoder(
                            input_dim=input_dim, 
                            hidden_dims=hidden_dims, 
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