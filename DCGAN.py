#Code was run on Google Colab

import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os

# Download the CelebA dataset using Kaggle API
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
print("Path to dataset files:", path)

# Hyperparameters
batch_size = 128  # Number of images per training batch
image_size = 64   # Image dimensions (64x64)
nz = 100  # Size of the latent vector (input to Generator)
ngf = 64  # Number of feature maps in the Generator
ndf = 64  # Number of feature maps in the Discriminator
epochs = 10  # Total number of training epochs
lr = 0.0002  # Learning rate
beta1 = 0.5  # Adam optimizer beta1 parameter

# Load CelebA dataset
# Transformations: Resize, Crop, Normalize to [-1, 1] (required for Tanh activation in Generator)
dataset = dset.ImageFolder(root="/root/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2/img_align_celeba",  
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# Create a DataLoader for batch processing
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Define the Generator Model
define_G = nn.Sequential(
    nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
    nn.BatchNorm2d(ngf * 8),
    nn.ReLU(True),
    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf * 4),
    nn.ReLU(True),
    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf * 2),
    nn.ReLU(True),
    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf),
    nn.ReLU(True),
    nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
    nn.Tanh()
)
generator = define_G

# Define the Discriminator Model
define_D = nn.Sequential(
    nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ndf * 2),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ndf * 4),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ndf * 8),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
    nn.Sigmoid()
)
discriminator = define_D

# Define loss function and optimizers
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Fixed noise for visualizing Generator progress
fixed_noise = torch.randn(64, nz, 1, 1)
g_losses, d_losses = [], []
print("Starting Training...")

# Training Loop
for epoch in range(epochs):
    lossD_epoch, lossG_epoch = 0, 0  # Track loss per epoch
    num_batches = len(dataloader)

    for i, data in enumerate(dataloader, 0):
        real_images = data[0]  # Load real images
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train Discriminator
        optimizerD.zero_grad()
        output_real = discriminator(real_images).view(-1, 1)
        lossD_real = criterion(output_real, real_labels)
        lossD_real.backward()
        
        noise = torch.randn(batch_size, nz, 1, 1)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach()).view(-1, 1)
        lossD_fake = criterion(output_fake, fake_labels)
        lossD_fake.backward()
        lossD = lossD_real + lossD_fake
        optimizerD.step()

        # Train Generator
        optimizerG.zero_grad()
        output = discriminator(fake_images).view(-1, 1)
        lossG = criterion(output, real_labels)
        lossG.backward()
        optimizerG.step()

        # Accumulate losses
        lossD_epoch += lossD.item()
        lossG_epoch += lossG.item()

    g_losses.append(lossG_epoch / num_batches)
    d_losses.append(lossD_epoch / num_batches)
    print(f"Epoch [{epoch+1}/{epochs}] Loss D: {lossD_epoch:.4f}, Loss G: {lossG_epoch:.4f}")

# Save the trained model
torch.save(generator.state_dict(), "dcgan_generator.pth")
torch.save(discriminator.state_dict(), "dcgan_discriminator.pth")

# Plot Loss Curves
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses, label="Generator")
plt.plot(d_losses, label="Discriminator")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Generate Final Sample Images
def generate_images():
    generator.load_state_dict(torch.load("dcgan_generator.pth"))
    generator.to(device)
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise).cpu()
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(fake_images, padding=2, normalize=True), (1, 2, 0)))
    plt.show()

generate_images()

