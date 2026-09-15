"""
hw4_code.py

Model definitions, training procedure, sampling, and latent-space
visualization for the HW4 DCGAN (102 Category Flower Dataset).

This module is designed to be SAFE TO IMPORT: importing it (e.g.
`from hw4_code import Generator`) only defines the classes/functions
below -- it does not mount Google Drive, extract any dataset, or
start training. All of that only happens if you run this file
directly (`python hw4_code.py`) with the dataset already available
locally, via the `main()` function at the bottom.
"""

import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------
BATCH_SIZE = 128
IMAGE_SIZE = 64
NZ = 100          # latent vector size
NUM_EPOCHS = 50
LR = 0.0002
BETA1 = 0.5


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class FlowersDataset(Dataset):
    """Loads the 102 Category Flower Dataset images using the
    category_to_images.json label file."""

    def __init__(self, image_dir, json_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        with open(json_path, 'r') as f:
            category_to_images = json.load(f)

        for label_str, img_list in category_to_images.items():
            label = int(label_str) - 1  # classes should start at 0
            for img_name in img_list:
                self.image_paths.append(os.path.join(self.image_dir, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------
# Weight initialization (standard DCGAN init)
# ---------------------------------------------------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(NZ, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# ---------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def train(dataloader, device, num_epochs=NUM_EPOCHS):
    """Trains the DCGAN and returns (netG, netD, G_losses, D_losses)."""
    netG = Generator().to(device)
    netG.apply(weights_init)

    netD = Discriminator().to(device)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

    G_losses = []
    D_losses = []

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # --- Train Discriminator ---
            netD.zero_grad()
            real_images = data[0].to(device)
            b_size = real_images.size(0)
            label = torch.full((b_size,), 1., dtype=torch.float, device=device)

            output = netD(real_images).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, NZ, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(0.)

            output = netD(fake_images.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizerD.step()

            # --- Train Generator ---
            netG.zero_grad()
            label.fill_(1.)
            output = netD(fake_images).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

    return netG, netD, G_losses, D_losses


# ---------------------------------------------------------------------
# Plotting / visualization helpers
# ---------------------------------------------------------------------
def plot_losses(G_losses, D_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def show_generated_images(netG, device, num_images=10):
    netG.eval()
    with torch.no_grad():
        fixed_noise = torch.randn(num_images, NZ, 1, 1, device=device)
        generated_images = netG(fixed_noise).detach().cpu()

    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Generated Flowers")
    plt.imshow(np.transpose(
        vutils.make_grid(generated_images, padding=2, normalize=True), (1, 2, 0)))
    plt.show()


def build_image_pool(netG, device, pool_size=64):
    netG.eval()
    with torch.no_grad():
        pool_noise = torch.randn(pool_size, NZ, 1, 1, device=device)
        pool_images = netG(pool_noise).detach().cpu()

    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.title("Image Pool for Latent Space Analysis")
    plt.imshow(np.transpose(
        vutils.make_grid(pool_images, padding=2, normalize=True), (1, 2, 0)))
    plt.show()

    return pool_noise, pool_images


def calculate_l2_distance(pool_noise, idx1, idx2):
    z1 = pool_noise[idx1].view(-1)
    z2 = pool_noise[idx2].view(-1)
    dist = torch.norm(z1 - z2, p=2).item()
    print(f"L2 Distance between vectors of images {idx1} and {idx2} is: {dist:.4f}")
    return dist


def plot_latent_space_pairs(pool_noise, similar_pairs, dissimilar_pairs):
    print("\n--- Calculating L2 Distances for Chosen Pairs ---")
    for p1, p2 in similar_pairs:
        calculate_l2_distance(pool_noise, p1, p2)
    for p1, p2 in dissimilar_pairs:
        calculate_l2_distance(pool_noise, p1, p2)

    all_indices = [idx for pair in similar_pairs + dissimilar_pairs for idx in pair]
    selected_z = pool_noise[all_indices].view(len(all_indices), -1).cpu().numpy()

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(selected_z)

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'purple']
    markers_sim = 'o'
    markers_dissim = 'X'

    for i, pair in enumerate(similar_pairs):
        idx1, idx2 = i * 2, i * 2 + 1
        plt.scatter(z_2d[idx1, 0], z_2d[idx1, 1], c=colors[i], marker=markers_sim, s=150,
                    label='Similar Pair' if i == 0 else "")
        plt.scatter(z_2d[idx2, 0], z_2d[idx2, 1], c=colors[i], marker=markers_sim, s=150)
        plt.plot([z_2d[idx1, 0], z_2d[idx2, 0]], [z_2d[idx1, 1], z_2d[idx2, 1]],
                 c=colors[i], linestyle='-', alpha=0.5)

    for i, pair in enumerate(dissimilar_pairs):
        offset = len(similar_pairs) * 2
        idx1, idx2 = offset + i * 2, offset + i * 2 + 1
        plt.scatter(z_2d[idx1, 0], z_2d[idx1, 1], c=colors[i], marker=markers_dissim, s=150,
                    label='Dissimilar Pair' if i == 0 else "")
        plt.scatter(z_2d[idx2, 0], z_2d[idx2, 1], c=colors[i], marker=markers_dissim, s=150)
        plt.plot([z_2d[idx1, 0], z_2d[idx2, 0]], [z_2d[idx1, 1], z_2d[idx2, 1]],
                 c=colors[i], linestyle=':', alpha=0.5)

    plt.title("Latent Space (z) Visualization of Selected Image Pairs (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # NOTE: update these paths to wherever you keep the dataset locally.
    images_directory = 'flowers_data/jpg'
    json_path = 'category_to_images.json'

    dataset = FlowersDataset(image_dir=images_directory, json_path=json_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Success! Loaded {len(dataset)} images.")

    netG, netD, G_losses, D_losses = train(dataloader, device)

    torch.save(netG.state_dict(), 'hw4_model.pkl')
    print("Model saved to hw4_model.pkl")

    plot_losses(G_losses, D_losses)
    show_generated_images(netG, device, num_images=10)

    pool_noise, _ = build_image_pool(netG, device, pool_size=64)

    similar_pairs = [(26, 27), (61, 62), (2, 40)]
    dissimilar_pairs = [(7, 48), (26, 34), (63, 60)]
    plot_latent_space_pairs(pool_noise, similar_pairs, dissimilar_pairs)


if __name__ == "__main__":
    main()