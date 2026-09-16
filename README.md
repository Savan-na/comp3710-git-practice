# COMP3710 Pattern Recognition and Analysis

Repository for COMP3710 Lab Demonstration work.

---

# Demo 2

## Part 1 - Discrete Fourier Transform

Implemented Fourier-series reconstruction and frequency-domain analysis of a square wave.

Key components:

- Square-wave reconstruction using odd harmonics
- Comparison with increasing numbers of harmonics
- Gibbs phenomenon near discontinuities
- Naive Discrete Fourier Transform
- FFT comparison
- PyTorch tensor implementation
- CUDA GPU implementation
- Runtime comparison

Main observation:

Increasing the number of harmonics improves the approximation away from discontinuities, but oscillations remain near sharp transitions due to the Gibbs phenomenon.

The naive DFT has approximately O(N^2) computational complexity, whereas the FFT reduces this to approximately O(N log N).

---

## Part 2 - Eigenfaces and Classification

Implemented face recognition using PCA eigenfaces and Random Forest classification on the LFW dataset.

Pipeline:

1. Load and split the LFW dataset
2. Subtract the training-set mean face
3. Apply SVD / PCA
4. Construct eigenfaces
5. Project faces into the PCA feature space
6. Visualise cumulative explained variance
7. Train a Random Forest classifier
8. Evaluate classification performance

PCA performs unsupervised dimensionality reduction, while the Random Forest performs supervised classification using the reduced PCA representation.

---

## Part 3 - Convolutional Neural Networks

### Part 3.1 - CNN Face Recognition

Implemented a CNN classifier using the LFW face dataset.

The network includes:

- Two convolutional layers
- 3 x 3 kernels
- 32 filters per convolutional layer
- Dense classification layers

Unlike PCA, the CNN learns hierarchical spatial features directly from image pixels.

---

## Part 3.2 - DAWNBench / ResNet-18 on CIFAR-10

Implemented a ResNet-18 classifier for CIFAR-10.

Final result:

- Best test accuracy: **94.87%**
- Mixed-precision training supported
- Rangpur GPU execution verified

Rangpur demonstration rehearsal:

- GPU: **NVIDIA A100-PCIE-40GB**
- CUDA available: **True**
- Test inference accuracy: **94.87%**
- Inference time: **5.07 s**
- One additional training epoch: **14.58 s**
- Job completed successfully with exit code `0:0`

The demo mode supports checkpoint loading, test inference, and one complete training epoch on Rangpur.

---

# Part 4 - Recognition Tasks

## Task 1 - Variational Autoencoder

Implemented a convolutional Variational Autoencoder using OASIS brain MRI slices.

### Architecture

Input:

`[B, 1, 256, 256]`

Encoder channel progression:

`1 -> 32 -> 64 -> 128 -> 256`

The encoder produces latent mean `mu` and latent log-variance `logvar`.

Latent dimension:

`2`

The latent sample is generated using the reparameterisation trick.

The decoder reconstructs the MRI image back to:

`[B, 1, 256, 256]`

### Loss

The VAE objective combines:

- reconstruction loss
- KL-divergence regularisation

### Final training result

Dataset sizes:

- Training: **9664**
- Validation: **1120**
- Test: **544**

Training:

- Epochs: **20**
- Best checkpoint epoch: **11**
- Validation total loss: **16869.5044**
- Reconstruction loss: **16861.4359**
- KL loss: **8.0687**

A two-dimensional latent manifold was successfully generated from the trained model.

Files:

- `demo2/part4/vae_oasis.py`
- `demo2/part4/vae_manifold.py`
- `demo2/part4/vae_train_job.sh`
- `demo2/part4/vae_manifold.png`

---

## Task 2 - U-Net Segmentation

Implemented U-Net semantic segmentation for OASIS brain MRI.

Raw segmentation labels are mapped as:

- `0 -> class 0`
- `85 -> class 1`
- `170 -> class 2`
- `255 -> class 3`

Input:

`[B, 1, 256, 256]`

Output:

`[B, 4, 256, 256]`

### Architecture

Encoder:

`1 -> 32 -> 64 -> 128 -> 256`

Bottleneck:

`256 -> 512`

Decoder:

`512 -> 256 -> 128 -> 64 -> 32`

Skip connections connect corresponding encoder and decoder stages.

### Loss

Training combines:

- Cross-Entropy loss
- Differentiable Dice loss

### Final test results

Best checkpoint:

- Epoch: **2**

Test loss:

- **0.1181**

Per-class Dice Similarity Coefficient:

- Class 0: **0.9972**
- Class 1: **0.9256**
- Class 2: **0.9288**
- Class 3: **0.9478**

All four segmentation classes achieved DSC greater than **0.90**.

Files:

- `demo2/part4/unet_oasis.py`
- `demo2/part4/unet_evaluate.py`
- `demo2/part4/unet_train_job.sh`
- `demo2/part4/unet_test_prediction.png`

---

## Task 3 - Generative Adversarial Network

Two GAN experiments were performed on OASIS brain MRI data.

### GAN V1

The first model generated 256 x 256 images using a deep convolutional generator.

Training completed successfully for 40 epochs, but visual inspection revealed:

- strong mode collapse
- highly similar generated samples
- checkerboard artefacts
- insufficient anatomical detail

The first experiment was retained as evidence of the model-development and debugging process.

Files:

- `demo2/part4/gan_oasis.py`
- `demo2/part4/gan_train_job.sh`
- `demo2/part4/gan_outputs/gan_final_generated_brains.png`
- `demo2/part4/gan_outputs/gan_training_losses.png`

### GAN V2 - WGAN-GP

The second GAN was redesigned to improve training stability, diversity and image quality.

Main changes:

- Reduced image resolution from **256 x 256** to **64 x 64**
- Replaced deep transposed convolutions with **nearest-neighbour upsampling followed by Conv2d**
- Replaced the original adversarial objective with **Wasserstein GAN with Gradient Penalty**
- Used multiple critic updates per generator update

Generator progression:

`latent vector -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64`

### Final training result

- Training epochs: **60**
- Training time: **5.69 minutes**
- Final critic loss: **-1.011358**
- Final generator loss: **0.158136**
- Final gradient penalty: **0.003547**
- Final generated-sample diversity measure: **7.835127**

The V2 model produced visibly more diverse brain MRI-like samples than V1.

Compared with V1:

- obvious mode collapse was substantially reduced
- checkerboard artefacts were substantially reduced
- brain boundaries became recognisable
- ventricular and internal structures became visible
- generated samples showed different anatomical appearances

The generated images are recognisably MRI-like, although fine anatomical detail remains limited by the reduced 64 x 64 image resolution.

Files:

- `demo2/part4/gan_oasis_v2.py`
- `demo2/part4/gan_v2_train_job.sh`
- `demo2/part4/gan_outputs_v2/real_oasis_64.png`
- `demo2/part4/gan_outputs_v2/generated_epoch_001.png`
- `demo2/part4/gan_outputs_v2/generated_epoch_020.png`
- `demo2/part4/gan_outputs_v2/generated_epoch_040.png`
- `demo2/part4/gan_outputs_v2/generated_epoch_060.png`
- `demo2/part4/gan_outputs_v2/gan_v2_final_generated_brains.png`
- `demo2/part4/gan_outputs_v2/gan_v2_training_losses.png`

The progression from epochs 1 to 60 provides visual evidence of the generator learning the OASIS MRI distribution.

---

# Rangpur Compute Environment

GPU jobs were submitted through Slurm using:

```text
--partition=comp3710
--account=comp3710
--gres=gpu:1
```

GPU workloads were executed successfully on NVIDIA A100 compute nodes.

The Rangpur login node was used only for login, file management and job submission.

---

# Git and Version Control

Large datasets, model checkpoints and Python cache files are excluded from version control.

Examples include:

- OASIS datasets
- CIFAR-10 datasets
- `*.pt` model checkpoints
- `__pycache__`
- `*.pyc`

Model checkpoints are retained locally / on Rangpur rather than committed to GitHub.

Completed Git courses:

- Introduction to Version Control with Git
- Version Control for Teams using Git

---

# AI Assistance

Generative AI was used as a development and learning assistant during parts of the implementation, debugging and documentation process.

AI assistance was used for tasks such as:

- discussing implementation approaches
- debugging code and environment issues
- reviewing model architectures
- interpreting training results
- improving documentation

All submitted code was executed and tested by the student, and the model architectures, training procedures, outputs and design decisions were reviewed and understood before demonstration.
