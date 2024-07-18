from torch import nn
import torch
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(AutoEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc#.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale#.cuda()
        self.kl = 0

    def forward(self, x):
        breakpoint()
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        
        logvar = self.linear3(x)
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        sigma = torch.exp(logvar / 2)
        
        z = mu + sigma * self.N.sample(mu.shape)
        
        self.kl = (sigma**2 + mu**2 - logvar - 1).sum() / 2
        
        self.mu = mu
        self.logvar = logvar
        
        return z

