import matplotlib.pyplot as plt
import numpy as np

def plot(train_dataset, n):
    
    fig, axs = plt.subplots(2, 4, figsize=(4, 2), facecolor='none')
        
    for i in range(2):
        for j in range(4):
            
            # get and rangom image and label
            rand = np.random.randint(0, len(train_dataset))
            image, label = train_dataset[rand]  # Get image and label
            image_numpy = image.numpy().squeeze()    # Convert image tensor to numpy array
            axs[i, j].imshow(image_numpy, cmap='gray')  # Plot the image
            axs[i, j].axis('off')  # Turn off axis
            axs[i, j].set_title(label_to_name(label), fontsize=8)  # Set title with item name
    plt.tight_layout()  # Adjust layout
    plt.savefig(f"./figs/plot{n}.png", dpi=300, bbox_inches='tight')  # Save as PNG with 300 DPI


def label_to_name(label):
    labels = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    return labels[label]