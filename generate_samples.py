import torch
import os
from dcgan_model import Generator

def reproduce_hw4():
    """
    Loads the trained GAN model and generates 1020 images as required.
    Expects 'generator_weights.pkl' to be in the same directory.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nz = 100
    num_classes = 102
    images_per_class = 10
    total_images = num_classes * images_per_class

    # אתחול המודל 
    netG = Generator().to(device)

    # טעינת המשקלים מהקובץ
    model_path = 'generator_weights.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weights file '{model_path}' not found in the current directory.")

    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()

    # ייצור התמונות מרעש אקראי
    noise = torch.randn(total_images, nz, 1, 1, device=device)

    with torch.no_grad():
        generated_images = netG(noise)

    return generated_images
