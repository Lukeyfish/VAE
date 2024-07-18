import torch
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import numpy as np


def plot_latent(autoencoder, data, num_batches=100, epoch=0, loss=0, bce=0, kld=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_dim = 784
    for i, (x, y) in enumerate(data):
        x = x.float() / 255.0  # Normalize manually to [0, 1]
        x = x.view(-1, x_dim)
        
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
        
    plt.title("Latent Space Representation", fontsize=26)
    plt.text(0.5, 1.05, f"Loss: {loss:.4f}  KLD: {kld:.4f}  BCE: {bce:.4f}", 
             horizontalalignment='center', 
             verticalalignment='center', 
             transform=plt.gca().transAxes, 
             fontsize=20, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlim(-0.6, 1.75)
    plt.ylim(-0.6, 1.75)
    
    plt.savefig(f"figs/latents/latent_{epoch}.png")
    plt.clf()

def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    #plt.imshow(img, extent=[*r0, *r1])
    plt.savefig('figs/reconstructed.png')
    plt.clf()
    
def plot_latent_space(model, scale=1.0, n=25, digit_size=28, figsize=15, epoch=0):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid 
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float)
            x_decoded = model.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = digit

    plt.figure(figsize=(figsize, figsize))
    #plt.imshow(figure, cmap="Greys_r")

    plt.title('VAE Latent Space Visualization')
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.savefig(f"figs/latent_space/latent_space_{epoch}.png")
    plt.clf()


def visualize_reconstructions(model, data_loader, device, epoch=0):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            images, _ = batch
            images = images.reshape(-1, 784)
            images = images.float() / 255.0
            images = images.to(device)
            recon_images, _, _ = model(images)
            
            # Detach and move to CPU for plotting
            images = images.cpu().numpy()
            recon_images = recon_images.cpu().numpy()
            
            fig, axs = plt.subplots(2, 5, figsize=(15, 6))
            fig.suptitle(f"Original and Reconstructed Images Epoch {epoch}", fontsize=22)

            # Plot original images
            for i in range(5):
                for j in range(5):
                    if i == 0:
                        axs[i, j].imshow(images[j].reshape(28, 28), cmap='gray')
                        axs[i, j].axis('off')
                        axs[i, j].set_title(f'Original {j+1}')

            # Plot reconstructed images
            for i in range(5):
                for j in range(5):
                    if i == 1:
                        axs[i, j].imshow(recon_images[j].reshape(28, 28), cmap='gray')
                        axs[i, j].axis('off')
                        axs[i, j].set_title(f'Reconstructed {j+1}')
            
            plt.savefig(f"figs/before&after/before&after_{epoch}.png")
            plt.clf()
            break
