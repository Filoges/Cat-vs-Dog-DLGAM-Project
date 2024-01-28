import torch
from torchvision.utils import save_image
import os
import json

import models

nz = 100

with open('samples_gen.json', 'r') as config_file:
    config = json.load(config_file)

generator_model_path = config['generator_model_path']
image_size = config['image_size']
generator_type = config['model_type']
num_images_to_generate = config['num_images_to_generate']
output_dir = config['output_folder']

# Instantiate your generator
if generator_type == "WGAN":
    if image_size == 32:
        generator = models.WGANGen32(3, 2)
    elif image_size == 64:
        generator = models.WGANGen64(3, 2)
    elif image_size == 128:
        generator = models.WGANGen128(3, 2)

elif generator_type == "CGAN":
    if image_size == 32:
        generator = models.CGANGen32(3, 2)
    elif image_size == 64:
        generator = models.CGANGen64(3, 2)
    elif image_size == 128:
        generator = models.CGANGen128(3, 2)

elif generator_type == "GAN":
    if image_size == 32:
        generator = models.Generator32(3)
    elif image_size == 64:
        generator = models.Generator64(3)
    elif image_size == 128:
        generator = models.Generator128(3)

# Load the saved model state_dict
generator.load_state_dict(torch.load(generator_model_path))

# Set the generator to evaluation mode
generator.eval()

# Define the output directory where images will be saved
os.makedirs(output_dir, exist_ok=True)

# Generate and save individual images
for i in range(num_images_to_generate):
    # Generate random noise tensor
    noise = torch.randn(1, 100, 1, 1)  # Assuming batch size 1 and noise dimension 100

    # Generate random labels
    if generator_type != "GAN":
        labels = torch.randn(1, 2, 1, 1)  # Assuming batch size 1 and label dimension 2

        # Generate an image using the generator
        with torch.no_grad():
            generated_image = generator(noise, labels)

    else:
        # Generate an image using the generator
        with torch.no_grad():
            generated_image = generator(noise)

    # Save the generated image to a file
    image_filename = os.path.join(output_dir, f'generated_image_{i}.png')
    save_image(generated_image, image_filename, normalize=True)

print(f'{num_images_to_generate} images generated and saved to {output_dir}')