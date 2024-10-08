import os
import torch
from torch import nn
from torch.functional import F
from tqdm import tqdm
import wandb

# Local Imports
from vis import plot_latent_space, plot_latent, visualize_reconstructions, sample_latent_points

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
        num_epochs,
        kl_annealing_cycles, #=1,  # or any other value you prefer
        kl_annealing_ratio, #=0.5  # or any other value you prefer
        sweep_id
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.save_name = save_name
        self.num_epochs = num_epochs
        self.kl_annealing_cycles = kl_annealing_cycles
        self.kl_annealing_ratio = kl_annealing_ratio
        self.sweep_id = sweep_id
    
    def train(self):
        
        autoencoder = self.model
        autoencoder.to(self.device)
        optimizer = torch.optim.Adam(autoencoder.parameters())
        x_dim = 784
        
        total_steps=0
        
        rate = 1
        
        self.model.train()
        
        best_val_rec_loss = float('inf')
        best_val_loss = float('inf')
        best_val_kld = float('inf')
        
        for epoch in range(self.num_epochs):
            total_reconstruction_loss = 0
            total_kld = 0
            total_loss = 0

            kl_weight = self.get_kl_weight(epoch)
            
            train_loader_tqdm = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=True)
            self.model.train()
            
            for step, (x, y) in enumerate(train_loader_tqdm):
                x = x.float() / 255.0  # Normalize manually to [0, 1]
                
                x = x.reshape(-1, 1, 28, 28).to(self.device)
                #x = x.reshape(-1, 784).to(self.device)
                

                optimizer.zero_grad()
                
                '''
                #GMM Prior
                recon_batch, mean, logvar, comp_logits, comp_means, comp_log_vars = self.model(x)
                loss = self.model.loss_function(recon_batch, x, mean, logvar, comp_logits, comp_means, comp_log_vars)
                
                reconstruction_loss = F.binary_cross_entropy(recon_batch, x, reduction='sum')
                kld = loss - reconstruction_loss  # The KLD is the difference between total loss and reconstruction loss
                
                weighted_loss = reconstruction_loss + kl_weight * kld
                
                weighted_loss.backward()
                '''
                
                #N(0, 1) prior
                x_hat, mean, log_var = self.model(x)
                reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
                kld = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())                
                loss = reconstruction_loss + kl_weight * kld
                
                
                loss.backward()
                
                optimizer.step()
                
                total_reconstruction_loss += reconstruction_loss.item()
                total_kld += kld
                total_loss += loss.item()

                
                train_loader_tqdm.set_postfix(loss=loss.item(), reconstruction_loss=reconstruction_loss.item(), kld=kld.item())
                
                
                if step % rate == 0:
                    plot_latent(autoencoder, self.val_loader, epoch=total_steps, rate=rate, loss=loss, rec_loss=reconstruction_loss, kld=kld)
                    #plot_latent_space(autoencoder, epoch=total_steps)
                    visualize_reconstructions(autoencoder, self.val_loader, self.device, epoch=total_steps)
                    #sample_latent_points(autoencoder, self.val_loader, self.device, total_steps)
                    
                if step % 30 == 0:
                    rate+=1
                
                total_steps+=1            
            
            avg_rec_loss = total_reconstruction_loss / len(self.train_loader.dataset)
            avg_kld = total_kld / len(self.train_loader.dataset)
            avg_loss = total_loss / len(self.train_loader.dataset)

            #visualize_reconstructions(autoencoder, self.val_loader, self.device, epoch=total_steps, sweep_id=self.sweep_id)

            
            #plot_latent_space(autoencoder, epoch=epoch)
            #plot_latent(autoencoder, self.val_loader, epoch=epoch, loss=avg_loss, bce=avg_bce, kld=avg_kld)
            
            print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}, Average Reconstruction Loss: {avg_rec_loss}, Average KLD: {avg_kld}')
            #wandb.log({"Epoch": epoch, "Average Loss": avg_loss, "Average Reconstruction Loss": avg_rec_loss, "Average KLD": avg_kld})
            
            val_rec_loss, val_kld, val_loss = self.validate()
            if(val_loss < best_val_loss):
                best_val_loss = val_loss
                best_val_rec_loss = val_rec_loss
                best_val_kld = val_kld
                print("Model Improved!")
            else:
                print("Model not improved!")
                #breakpoint()
            
            
        return autoencoder
    
    
    def get_kl_weight(self, epoch):
        cycle_length = self.num_epochs // self.kl_annealing_cycles
        cycle = epoch // cycle_length
        cycle_progress = (epoch % cycle_length) / (cycle_length * self.kl_annealing_ratio)
        kl_weight = min(1.0, cycle_progress + cycle)
        return kl_weight
    
                
    def validate(self):
        val_loader_tqdm = tqdm(self.val_loader, desc=f"Evaluating", leave=True)
        
        self.model.eval()
        
        val_rec_loss = 0.0
        val_kld = 0.0
        val_loss = 0.0

        kl_weight = self.get_kl_weight(self.num_epochs - 1)  # Use the final KL weight for validation
        
        for step, (x, y) in enumerate(val_loader_tqdm):
            x = x.float() / 255.0  # Normalize manually to [0, 1]
            
            x = x.reshape(-1, 1, 28, 28).to(self.device)
            
            #x = x.view(-1, 784).to(self.device)
            
            '''
            #GMM Prior
            recon_batch, mean, logvar, comp_logits, comp_means, comp_log_vars = self.model(x)
            loss = self.model.loss_function(recon_batch, x, mean, logvar, comp_logits, comp_means, comp_log_vars)
            
            reconstruction_loss = F.binary_cross_entropy(recon_batch, x, reduction='sum')
            kld = loss - reconstruction_loss  # The KLD is the difference between total loss and reconstruction loss
            '''
            
            # N(0, 1) prior
            x_hat, mean, log_var = self.model(x)
            
            reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
            kld = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            loss = reconstruction_loss + kl_weight * kld
            
            
            val_rec_loss += reconstruction_loss.item()
            val_kld += kld
            val_loss += loss.item()

            
            val_loader_tqdm.set_postfix(loss=loss.item(), reconstruction_loss=reconstruction_loss.item(), kld=kld.item())
            
        avg_val_rec_loss = val_rec_loss / len(self.val_loader.dataset)
        avg_val_kld = val_kld / len(self.val_loader.dataset)
        avg_val_loss = val_loss / len(self.val_loader.dataset)

            
        print(f'Validation Average Loss: {avg_val_loss}, Average Reconstruction Loss: {avg_val_rec_loss}, Average KLD: {avg_val_kld}')
        #wandb.log({"Validation Average Loss": avg_val_loss, "Validation Average Reconstruction Loss": avg_val_rec_loss, "Validation Average KLD": avg_val_kld})
        return avg_val_rec_loss, avg_val_kld, avg_val_loss

    
    
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
