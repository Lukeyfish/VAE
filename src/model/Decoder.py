import torch
from torch import nn
import torch.nn.functional as F

class AutoDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(AutoDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 28, 28))

