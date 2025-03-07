
# Deep Convolutional Generative Adversarial Network (DCGAN) - CelebA Dataset

## Overview

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic face images. The model is trained on the CelebA Faces dataset using PyTorch.

## Dataset Preprocessing

1. Download Dataset: The CelebA dataset is downloaded using kagglehub.

2. Transformation Pipeline:

* Resize images to 64x64 pixels.

* Center crop the images.

* Convert images to tensor format.

* Normalize pixel values to the range [-1, 1] for stable training.

## Model Architecture

1. Generator:

* Takes a 100-dimensional latent vector as input.
* Uses transposed convolution layers to upsample the noise into a 64x64 RGB image.
* Includes batch normalization and ReLU activations.
* Outputs an image with a tanh activation function.

2. Discriminator:

* Takes a 64x64 image as input.
* Uses convolutional layers with leaky ReLU activations.
* Includes batch normalization to stabilize training.
* Outputs a probability score using a sigmoid activation function.

## Training Instructions

* Install dependencies and run the code.
* The model trains for 10 epochs by default.
* Generator and Discriminator losses are tracked.
* Generated images are saved after each epoch.
* Loss curves are plotted at the end of training.
* Generated images can be viewed to evaluate performance.
