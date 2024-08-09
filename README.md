# Variational Auto Encoder

A Variational Autoencoder is a type of generative model that learns to map data to a latent space which is later used to reconstruct images. (see Figure 1)
This mapping of images to a latent space allows the model to learn quickly and sample directly from the latent space to generate new data points. (see Figure 
I used this repository as practice to understand how Variational Autoencoders effect the representational power of the data and the fidelity the reconstructed images hold.
VAE's are integral to other types of generative models such as GAN's or Latent Diffusion Models.  


As seen in Figure 1, the original image is sent through an encoder which maps the original image to the probabilistic latent space. Once here the image can be reconstructed from the latent space via the decoder.  

<div align="center">
  <img align="center" src="https://github.com/Lukeyfish/VAE/blob/main/imgs/VAE_Diagram.png" alt="VAE_Diagram">
  <p style="text-align: center;"> 
    Figure 1: VAE Architecture Diagram
  </p>
</div>


Figure 2 highlights the learned Guassian latent space via the learned mean and variance. Slight alterations in the sampled mean or var result in closer alignments of other objects. 

<div align="center">
  <img align="center" height=500, src="https://github.com/Lukeyfish/VAE/blob/main/imgs/Latent_Space.png" alt="Latent_Space">
  <p>Figure 2: Visualization of the Latent Space</p>
</div>



## Training

To help me understand how the model was learning to differente between classes, I created some helpful gifs.  
Figure 3 is showing the Latent Space representation of the data once it has been passed through the encoder. (Notice how seperated they get towards the end?!)  
Figure 4 is showing the learning process of the VAE overtime. As you can see as time evolves the reconstructed images slowly align closer to the original images yay! 🎉
<div align="center">
  <table>
    <tr>
      <td>
        <img height=400, src="https://github.com/Lukeyfish/VAE/blob/main/imgs/Latent_Representation.gif" alt="Latent_Representation">
        <p align="center">Figure 3: Visualization of the latent space learning the encoder mappings</p>
      </td>
      <td>
        <img height=400, src="https://github.com/Lukeyfish/VAE/blob/main/imgs/Latent_Progression.gif" alt="Latent_Progression" width="400">
        <p align="center">Figure 4: VAE learning to reconstruct images</p>
      </td>
    </tr>
  </table>
</div>





## Installation

To install the required dependencies, run the following command:

```bash
pip install -r environment.yml
```

To download fashion-mnist data run:

```bash
make data
```

## Configuration

Edit the `configs/main.yaml` with data load path, and desired hyperparameters

## Usage

To run the model:

```bash
Make train
```


