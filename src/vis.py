import torch
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import numpy as np


def plot_latent(autoencoder, data, num_batches=100, epoch=0, rate=1, loss=0, rec_loss=0, kld=0):
    elevation = 30
    azimuths = [30, 60, 120, 240]  # Different azimuth angles for the subplots
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_dim = 784
    
    fig = plt.figure(figsize=(14, 10))  # Adjust figure size as needed
    fig.suptitle("Latent Space Representation (Step {}, Loss {:.4f}, Reconstruction Loss {:.4f}, KLD {:.4f})".format(epoch, loss, rec_loss, kld), fontsize=16)
    
    for idx, azimuth in enumerate(azimuths, 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')  # Create 3D axes for each subplot
        
        for i, (x, y) in enumerate(data):
            x = x.float() / 255.0  # Normalize manually to [0, 1]
            x = x.view(-1, x_dim)
            
            z = autoencoder.encoder(x.to(device))
            z = z.to('cpu').detach().numpy()
            
            ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=y, cmap='tab10')  # Scatter plot in 3D
            if i > num_batches:
                break
        
        ax.set_xlabel('Z1')
        ax.set_ylabel('Z2')
        ax.set_zlabel('Z3')
        ax.view_init(elev=elevation, azim=azimuth)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust top spacing for the suptitle
    
    '''
    plt.text(0.5, 1.05, f"Loss: {loss:.4f}  KLD: {kld:.4f}  BCE: {bce:.4f}", 
             horizontalalignment='center', 
             verticalalignment='center', 
             transform=plt.gca().transAxes, 
             fontsize=7, )
             #bbox=dict(facecolor='white', alpha=0.8))
    '''
    
    #plt.xlim(-0.6, 1.75)
    #plt.ylim(-0.6, 1.75)
    
    plt.savefig(f"figs/latents/latent_{epoch}.png", dpi=70)
    plt.clf()
    plt.close(fig)

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


def visualize_reconstructions(model, data_loader, device, epoch=0, sweep_id=None):
    model.eval()
    class_examples = {i: None for i in range(10)}  # Dictionary to hold one example from each class

    # Find one example of each class
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            for i in range(10):
                if class_examples[i] is None:
                    idx = (labels == i).nonzero(as_tuple=True)[0]
                    if len(idx) > 0:
                        class_examples[i] = images[idx[0]].reshape(-1, 784).float() / 255.0
            if all(v is not None for v in class_examples.values()):
                break

    if not all(v is not None for v in class_examples.values()):
        raise ValueError("Not all classes are represented in the data loader.")

    images = torch.stack(list(class_examples.values()))
    images = images.to(device)
    images = images.reshape(-1, 1, 28, 28)
    recon_images, _, _= model(images)

    # Detach and move to CPU for plotting
    images = images.cpu().numpy()
    recon_images = recon_images.detach().numpy()

    fig, axs = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle(f"Original and Reconstructed Images. Step: {epoch}", fontsize=22)

    # Plot original and reconstructed images in 4x5 grid
    for i in range(10):
        row, col = divmod(i, 5)
        axs[row * 2, col].imshow(images[i].reshape(28, 28), cmap='gray')
        axs[row * 2, col].axis('off')
        axs[row * 2, col].set_title(f'Original {i}')
        
        axs[row * 2 + 1, col].imshow(recon_images[i].reshape(28, 28), cmap='gray')
        axs[row * 2 + 1, col].axis('off')
        axs[row * 2 + 1, col].set_title(f'Reconstructed {i}')
    
    plt.savefig(f"figs/before&after/before&after_{sweep_id}_{epoch}.png", dpi=80)
    plt.close(fig)