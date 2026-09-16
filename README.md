# COMP3710 Pattern Recognition and Analysis

Repository for COMP3710 Lab Demonstration work.

## Demo 2

### Part 1 - Discrete Fourier Transform
- Square-wave Fourier reconstruction
- Gibbs phenomenon
- DFT frequency decomposition
- PyTorch / GPU implementation

### Part 2 - Eigenfaces and Classification
- PCA and eigenfaces
- Low-dimensional face representation
- Random Forest classification

### Part 3 - Convolutional Neural Networks

#### CNN
Implemented CNN-based face recognition.

#### DAWNBench
Implemented ResNet-18 for CIFAR-10.

Best local test accuracy: **94.87%**

The demo mode supports checkpoint loading, inference, and one complete training epoch on Rangpur.

### Part 4 - Recognition

#### Task 1 - Variational Autoencoder
Implemented a convolutional VAE for OASIS brain MRI.

Key features:
- 1 x 256 x 256 grayscale MRI input
- Convolutional encoder and decoder
- 2D latent space
- Mean and log-variance prediction
- Reparameterisation trick
- Reconstruction + KL-divergence loss
- Latent manifold visualisation

Files:
- `demo2/part4/vae_oasis.py`
- `demo2/part4/vae_manifold.py`
- `demo2/part4/vae_train_job.sh`

Final training result: pending Rangpur run.

#### Task 2 - U-Net Segmentation
Implemented U-Net semantic segmentation for OASIS MRI.

Raw mask labels are mapped as:
- 0 -> class 0
- 85 -> class 1
- 170 -> class 2
- 255 -> class 3

Input: `[B, 1, 256, 256]`
Output: `[B, 4, 256, 256]`

The implementation includes:
- U-Net encoder-decoder architecture
- Skip connections
- Cross-entropy and Dice loss
- Per-class DSC evaluation
- Test inference
- Segmentation visualisation

Files:
- `demo2/part4/unet_oasis.py`
- `demo2/part4/unet_evaluate.py`
- `demo2/part4/unet_train_job.sh`

Final DSC results: pending Rangpur training and evaluation.

## Rangpur

GPU jobs are submitted using the COMP3710 Slurm account:

`--account=comp3710`

## Git

Large datasets, model checkpoints and Python cache files are excluded from version control.

Completed Git courses:
- Introduction to Version Control with Git
- Version Control for Teams using Git
