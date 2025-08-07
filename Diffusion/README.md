# Diffusion Model

This project implements a denoising diffusion probabilistic model (DDPM) using a U-Net-based backbone with attention and PyTorch optimization techniques. The model learns to reverse a noising process to generate high-fidelity images from pure Gaussian noise.

---

## Overview

Diffusion models are generative models that learn to denoise random noise across a series of time steps. This project focuses on an image-space diffusion process using a U-Net with attention and timestep embeddings. The implementation is written in PyTorch with several performance improvements like ahead-of-time (AOT) compilation, fused optimizers, and automatic mixed precision (AMP).

---

## Methodology

The model is trained to predict the noise added to a clean image at a given timestep. The forward process (adding noise) is fixed, and the model learns the reverse process (denoising) by minimizing a simplified variational loss (mean squared error between predicted and actual noise).

---

## Model Architecture

* **U-Net Backbone**: Symmetric encoder-decoder with skip connections.
* **Residual Blocks**: Used in both encoder and decoder paths.
* **Multi-Head Self-Attention (MHSA)**: Injected at chosen spatial resolutions.
* **Timestep Embeddings**: Injected into each residual block for temporal conditioning.
* **Noise-Level Conditioning**: Integrated into the model via learned embeddings.

---

## Optimization Techniques

This implementation incorporates multiple optimization strategies:

* **`torch.compile`**: Entire model compiled using PyTorch’s ahead-of-time (AOT) compiler for reduced Python overhead and kernel fusion. Results in notable speedups during training and inference.
* **Fused Optimizers**: Uses fused versions of the Adam optimizer for lower-latency gradient updates.
* **Automatic Mixed Precision (AMP)**: Mixed-precision training via `torch.amp` to reduce memory consumption and accelerate training on modern GPUs.
* **Exponential Moving Average (EMA)**:

  * EMA is applied to model weights during training to stabilize sampling and improve generalization.
  * EMA weights are maintained separately and used during inference.
  * Helps reduce artifacts and improves output quality in later stages of training.
* **Efficient Scheduling**:
  * Used CosineAnnealingLR
  * Cosine beta noise schedule for smooth forward process

---

## Training Strategy

* **Objective**: Mean Squared Error (MSE) between predicted and actual noise.
* **Optimizer**: AdamW, β1=0.9, β2=0.999.
* **Learning Rate**: Warmup followed by cosine annealing.
* **Batch Size**: A batch size of 24 was used, limited by GPU memory.
* **Training Duration**: Approximately 15 minutes per epoch on an RTX 4050 laptop GPU. Stable training achieved after extensive hyperparameter tuning.

---

## Sampling Process(DDIM-sampling)

At inference time:

1. Start from a pure Gaussian noise sample.
2. Apply the trained model iteratively to predict and remove noise across timesteps.
3. Parameters (e.g., number of steps, `eta`) affect diversity vs. fidelity.

---

## Observations & Results

* During training, checkerboard patterns and "void face" artifacts were observed when using perceptual loss, primarily due to an overly high perceptual loss weight.
* Switching to MSE loss led to more stable training dynamics and consistently produced sharper, artifact free outputs.
* Performance improved further by reducing the perceptual loss weight.
* Incorporating attention layers greatly enhanced the model's ability to maintain global consistency in generated images.
* Leveraging `torch.compile` and AMP resulted in a noticeable 35-40% improvement in training speed compared to standard FP32 precision.

---

## Future Work

* Experiment with VQ embeddings or latent-space diffusion.
* Explore image-to-image and text-to-image tasks.

---

## References

* Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020
* Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, ICML 2021
* PyTorch 2.0 documentation (for `torch.compile` and AMP)
