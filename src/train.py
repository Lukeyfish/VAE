import torch
import yaml
from torch.utils.data import Subset
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import numpy as np
import wandb

# Local Imports
from model.convnet import VGG11, VGG16, BasicCNN
from data import FashionDataset, FashionDataLoader
from trainer import Trainer, Tester
from model.VAE import VAE, VAE1
from model.VAEGMM import VAEGMM
from model.VAE2 import VAE2

from vis import visualize_reconstructions

sweep_config = {
    'method': 'random',  # or 'grid' or 'bayes'
    'metric': {
        'name': 'Validation Average Loss',
        'goal': 'minimize'   
    },
    'parameters': {
        'lr': {
            'values': [0.01, 0.001, 0.0001]
        },
        'kl_annealing_cycles': {
            'values': [1, 2, 3]
        },
        'kl_annealing_ratio': {
            'values': [0.5, 0.7, 0.9]
        },
        'dropout_rate': {
            'values': [0.0, 0.3, 0.5, 0.7]
        },
        'latent_dim': {
            'values': [2, 5, 20, 50, 100]
        },
        'hidden_dim': {
            'values': [200, 300, 400]
        }
    }
}


def main():
    '''
    run = wandb.init(
            project="VAE",
            )
    '''
    # Assigns training to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('training on: ', device)

    # Opens the train config
    with open('./configs/main.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
        
    '''
    # Sweeping over the hyperparameters
    config = wandb.config

    cfg['model']['dropout_rate'] = config.dropout_rate
    cfg['trainer']['optim']['lr'] = config.lr
    '''


    # Instantiating the Train and Test sets
    train_set = FashionDataset(cfg['data']['train_ft'], cfg['data']['train_tg'])

    # Splitting the training set into training and validation sets
    train_indices = list(range(0, 50000))
    val_indices = list(range(50000, len(train_set)))

    val_set = Subset(train_set, val_indices)
    train_set = Subset(train_set, train_indices)

    test_set = FashionDataset(cfg['data']['test_ft'], cfg['data']['test_tg'])
    

    # Instantiating the Model
    #latent_dim = config.latent_dim
    #hidden_dim = config.hidden_dim
    latent_dim = 50
    hidden_dim = 100
    #model = VAE(784, 400, latent_dims, device)
    #model = VAE1(784, hidden_dim, latent_dim, cfg['model']['dropout_rate'])
    
    model = VAE2(latent_dim=latent_dim, device=device).to(device)
    #model = VAEGMM(784, 400, latent_dim, cfg['model']['dropout_rate'])
    #model = LearnedPriorVAE(784, 400, latent_dim, cfg['model']['dropout_rate'])
    
    #model = BasicCNN(cfg['model']['dropout_rate']).to(device)
    
    # Instantiating the Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['trainer']['optim']['lr'])
    
    train_loader = FashionDataLoader(train_set, cfg['trainer']['batch_size'], cfg['trainer']['shuffle'])
    val_loader = FashionDataLoader(val_set, cfg['trainer']['batch_size'], False)
    test_loader = FashionDataLoader(test_set, cfg['trainer']['batch_size'], cfg['trainer']['shuffle'])
        
    # Instantiating the Trainer and Tester
    trainer = Trainer(
            train_loader.load(), 
            val_loader.load(), 
            model, 
            optimizer, 
            device, 
            cfg['model']['save_dir'],
            cfg['model']['save_name'],
            cfg['trainer']['num_epochs'],
            kl_annealing_cycles=2,
            kl_annealing_ratio=0.5,
            sweep_id=0
        )
    #tester = Tester(test_loader.load(), model, device)
    
    # Training the VAE
    vae = trainer.train()
    
    #visualize_reconstructions(vae, test_loader.load(), device)


    

if __name__=="__main__":
    main()


    #sweep_id = wandb.sweep(sweep_config, project="VAE", entity="lukedoj")
    #wandb.agent(sweep_id, function=main, count=50)
