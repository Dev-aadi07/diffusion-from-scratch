# Diffusion From Scratch

Learning how diffusion models work by building everything from scratch on CPU.

## Goal

Understand image generation deeply instead of using APIs or prebuilt models.

The focus is on intuition, experimentation, and gradually building toward a tiny diffusion model.

---

## Progress

### Step 00 - Python Setup

- Virtual environment
- Python basics
- Project structure

### Step 01 - Images as Arrays

- Grayscale images
- Pixels and brightness
- Displaying images with Matplotlib

### Step 02 - RGB Images

- RGB channels
- Color representation
- Random image generation

### Step 03 - Forward Diffusion

- Adding Gaussian noise
- Understanding diffusion intuition
- Gradual corruption of images

### Step 04 - First Neural Network

- Linear layers
- Tensors
- Loss functions
- Training basics

### Step 05 - Dataset Training

- Generated square datasets
- First denoising experiments
- Understanding memorization vs learning

### Step 06 - Autoencoder

- Image compression and reconstruction
- Denoising autoencoder
- Limitations of fully connected networks

### Step 07 - CNN Denoiser

- First convolutional neural network
- Learning image structure
- Better denoising performance

### Step 08 - Noise Prediction

- Predicting noise instead of clean images
- Diffusion-style denoising
- Noise subtraction

### Step 09 - Multi-Step Diffusion (In Progress)

- Multiple noise levels
- Training on varying corruption strengths
- Moving toward real diffusion pipelines

---

## Tech Stack

- Python
- NumPy
- PyTorch
- Matplotlib

---

## Hardware

Built and trained entirely on CPU.

No APIs.
No paid tools.
No pretrained models.

Everything is implemented from scratch for learning purposes.