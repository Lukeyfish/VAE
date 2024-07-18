import torch
from torch import nn
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    def __init__(
        self, 
        latent_dim,
        input_dim,
        hidden_dim,
        dropout_rate
        ):
        
        super(Encoder, self).__init__()
        
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

class Decoder(nn.Module):
    def __init__(
        self, 
        latent_dim,
        input_dim,
        hidden_dim,
        dropout_rate
        ):
        
        super(Decoder, self).__init__()
        
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
        
class VAEGMM(nn.Module):
    
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        latent_dim,
        dropout_rate,
        n_components=10 # Number of componenets for GMM
        ):
        
        super(VAEGMM, self).__init__()

        # encoder
        self.encoder = Encoder(
                            input_dim=input_dim, 
                            hidden_dim=hidden_dim, 
                            latent_dim=latent_dim, 
                            dropout_rate=dropout_rate
                            )
        
        self.decoder = Decoder(
                            input_dim=input_dim, 
                            hidden_dim=hidden_dim, 
                            latent_dim=latent_dim, 
                            dropout_rate=dropout_rate
                            )
        
        # Parameters for the GMM prior
        self.n_components = n_components
        self.component_logits = nn.Parameter(torch.zeros(n_components))
        self.component_means = nn.Parameter(torch.randn(n_components, latent_dim))
        self.component_log_vars = nn.Parameter(torch.zeros(n_components, latent_dim))
        
    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def log_normal(self, x, mean, logvar):
        return -0.5 * torch.sum(logvar + (x - mean).pow(2) / logvar.exp() + math.log(2 * math.pi), dim=1)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar, self.component_logits, self.component_means, self.component_log_vars

    
    def log_normal_mixture(self, z, component_logits, component_means, component_log_vars):
        z = z.unsqueeze(1)  # [B, 1, D]
        component_means = component_means.unsqueeze(0)  # [1, C, D]
        component_log_vars = component_log_vars.unsqueeze(0)  # [1, C, D]
        
        log_probs = -0.5 * torch.sum(
            component_log_vars + 
            (z - component_means).pow(2) / component_log_vars.exp() + 
            math.log(2 * math.pi),
            dim=2
        )  # [B, C]
        
        return torch.logsumexp(log_probs + component_logits, dim=1)
    
    def loss_function(self, recon_x, x, z_mean, z_logvar, component_logits, component_means, component_log_vars):
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence
        z = self.reparameterization(z_mean, z_logvar)
        log_q_z = self.log_normal(z, z_mean, z_logvar)
        log_p_z = self.log_normal_mixture(z, component_logits, component_means, component_log_vars)
        kl_loss = log_q_z - log_p_z
        
        return recon_loss + kl_loss.sum()