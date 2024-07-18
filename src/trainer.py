import os
import torch
from torch import nn
from torch.functional import F
from tqdm import tqdm

# Local Imports
from vis import plot_latent_space, plot_latent, visualize_reconstructions

# Trainer class to train the model and evaluate it
class Trainer:
    def __init__(
        self, 
        train_loader,
        val_loader, 
        model, 
        optimizer,
        device,
        save_path,
        save_name,
        num_epochs
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.save_name = save_name
        self.num_epochs = num_epochs
    
    def train(self):
        autoencoder = self.model
        autoencoder.to(self.device)
        optimizer = torch.optim.Adam(autoencoder.parameters())
        x_dim = 784
        
        self.model.train()
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            total_bce = 0
            total_kld = 0
            
            train_loader_tqdm = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=True)
            for x, y in train_loader_tqdm:
                x = x.float() / 255.0  # Normalize manually to [0, 1]
                x = x.view(-1, x_dim).to(self.device)
                

                optimizer.zero_grad()
                x_hat, mean, log_var = self.model(x)
                loss = self.loss_function(x, x_hat, mean, log_var)
                bce = F.binary_cross_entropy(x, x_hat, reduction='sum').item()
                kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()).item()
                
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_bce += bce
                total_kld += kld
                
                train_loader_tqdm.set_postfix(loss=loss.item(), bce=bce, kld=kld)
            
            
            
            avg_loss = total_loss / len(self.train_loader.dataset)
            avg_bce = total_bce / len(self.train_loader.dataset)
            avg_kld = total_kld / len(self.train_loader.dataset)
            
            #plot_latent_space(autoencoder, epoch=epoch)
            #plot_latent(autoencoder, self.train_loader, epoch=epoch, loss=avg_loss, bce=avg_bce, kld=avg_kld)
            visualize_reconstructions(autoencoder, self.val_loader, self.device, epoch=epoch)
            
            print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}, Average BCE: {avg_bce}, Average KLD: {avg_kld}')
        return autoencoder
    
    def loss_function(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD
    
                
    def validate(self):
        # Set the model to evaluation mode
        self.model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = nn.CrossEntropyLoss()(y_hat, y)
                val_loss += loss.item()
                
                _, predicted = torch.max(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        val_accuracy = 100 * correct / total
        
        return avg_val_loss, val_accuracy
    
    
class Tester:
    def __init__(
        self, 
        test_loader, 
        model, 
        device
    ):
        self.test_loader = test_loader
        self.model = model
        self.device = device
    
    def test(self):
        # Set the model to evaluation mode
        self.model.eval()
        
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                _, predicted = torch.max(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        test_accuracy = 100 * correct / total
        
        return test_accuracy