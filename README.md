# DCGAN Flower Image Generation & Latent Space Analysis

This repository contains a PyTorch implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) trained to generate images of flowers[cite: 20, 23]. The model is trained on the 102 Category Flower Dataset[cite: 20].

## 🧠 Project Overview
* **Architecture:** Unconditional DCGAN implementation mapping a 100-dimensional latent vector (z) to a 64x64 RGB image[cite: 20, 23].
* **Latent Space Analysis:** Evaluated the meaningfulness of the learned representations by calculating and comparing L2 norm distances between latent vectors of visually similar and dissimilar generated image pairs[cite: 20, 23].
* **Visualization:** Utilized Principal Component Analysis (PCA) to project and visualize the latent space manifold in 2D[cite: 20, 23].

## 📂 Repository Contents
* `dcgan_model.py`: Core architecture defining the Generator and Discriminator, alongside the training loop, weight initialization, and latent-space visualization functions[cite: 20].
* `generate_samples.py`: Inference script that loads the trained model weights and autonomously generates a batch of 1,020 images from random noise[cite: 21].
* `DCGAN_Latent_Space_Analysis.pdf`: Comprehensive analytical report detailing the training dynamics (e.g., minimax equilibrium, mode collapse observations) and the PCA latent space evaluation[cite: 23].
* `generator_weights.pkl`: Saved state dictionary of the trained Generator[cite: 21, 22].
