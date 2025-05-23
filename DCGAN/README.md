# Anime Face Generation with Deep Convolutional GAN

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate anime-style face images using PyTorch. Trained on the Anime Face Dataset, the model generates high-quality 128x128 image outputs.

---

## Overview

Generative Adversarial Networks (GANs) synthesize realistic data through adversarial training. This DCGAN generates anime-style faces using a generator and discriminator, trained on \~63,000 images from the Anime Face Dataset. The implementation uses PyTorch with optimizations for efficiency and stability.

---

## Methodology

The DCGAN consists of:

- **Generator**: Maps a 256-dimensional noise vector to a 128x128 RGB image using transposed convolutional layers.
- **Discriminator**: Classifies images as real or fake using convolutional layers.

Training uses binary cross-entropy loss, with the generator aiming to fool the discriminator and the discriminator distinguishing real from fake images.

### Dataset

- **Anime Face Dataset**: \~63,000 images, sourced via KaggleHub.
- **Preprocessing**: Resized to 128x128, normalized (mean=0.5, std=0.5).

---

## Model Architecture

- **Generator**: Transposed convolutional layers with batch normalization and ReLU activations.
- **Discriminator**: Convolutional layers with batch normalization and LeakyReLU activations.

---

## Implementation Details

### Training Strategy

- **Optimizer**: Adam.
- **Training Loop**: Trained for 100 epochs.
- **Loss Monitoring**: Tracks generator and discriminator losses.

## Results and Observations

- **Visual Quality**: Produces recognizable anime-style faces with minor checkerboard artifacts.
- **Stability**: Batch normalization and low learning rate reduce mode collapse.

---

## Challenges and Limitations

- **Checkerboard Artifacts**: Likely due to transposed convolutions with larger kernel size.
- **Mode Collapse**: Present earlier but mitigated through hyperparameter tuning.
- **Evaluation**: Lacks quantitative metrics like FID.

---

## Future Work

- Explore advanced GAN variants (e.g. StyleGAN).
- Implement FID and Inception Score for evaluation.
- Add conditional generation for style control.

---

## Conclusion

This DCGAN implementation generates anime-style faces with stable training and coherent outputs. Future enhancements in architecture and evaluation could further improve performance.

---

## References

- Goodfellow, I., et al. (2014). *Generative Adversarial Nets*. NeurIPS 2014.
- Radford, A., et al. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. arXiv:1511.06434.
- PyTorch Documentation (https://pytorch.org/docs/stable/index.html).
