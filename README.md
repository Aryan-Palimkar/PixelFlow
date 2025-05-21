# Anime Face Generation Project

This project explores generative modeling for anime-style face generation using two approaches: a Deep Convolutional Generative Adversarial Network (DCGAN) and a Denoising Diffusion Probabilistic Model (DDPM). Implemented in PyTorch, both models are trained on the Anime Face Dataset to produce 128x128 pixel images.

## Project Structure

- **DCGAN**: Implements a DCGAN for generating anime faces via adversarial training. See [DCGAN/README.md](DCGAN/README.md) for details.
- **Diffusion**: Implements a DDPM with a U-Net backbone for denoising-based image generation. See [Diffusion/README.md](Diffusion/README.md) for details.

## Dataset

- **Anime Face Dataset**: ~63,000 anime-style face images, sourced via KaggleHub, preprocessed to 128x128 resolution with normalization (mean=0.5, std=0.5).
## Notebooks and Pretrained Models

- **DCGAN Notebook**: The implementation and training process for the DCGAN model are provided in a Jupyter Notebook. It includes data preprocessing, model architecture, training loop, and sample generation.
- **Diffusion Notebook**: The DDPM implementation, training, and sampling process are detailed in a Jupyter Notebook. It also includes the use of Exponential Moving Average (EMA) weights for improved sampling quality.
- **Pretrained Weights**: 
    - DCGAN: Includes the trained generator weights for generating anime faces.
    - Diffusion: Includes both the model weights and EMA weights for high-quality image generation.

## How to Use

1. Open the respective Jupyter Notebook (`DCGAN.ipynb` or `Diffusion.ipynb`) to explore the implementation and training process.
2. Use the provided pretrained weights to generate anime faces without retraining the models.