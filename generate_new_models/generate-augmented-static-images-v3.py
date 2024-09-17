from PIL import Image, ImageEnhance, ImageOps
import random
import os
import numpy as np
import os

# Ensure the current working directory is the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_border_average_color(image):
    image = image.convert("RGB")
    np_image = np.array(image)
    top_border = np_image[0, :]
    bottom_border = np_image[-1, :]
    left_border = np_image[:, 0]
    right_border = np_image[:, -1]
    border_pixels = np.concatenate((top_border, bottom_border, left_border, right_border), axis=0)
    average_color = tuple(np.mean(border_pixels, axis=0).astype(int))
    return average_color

def apply_perspective_transform(image):
    width, height = image.size

    # Randomize the perspective points within a defined range
    left_shift = random.uniform(-0.2, 0.2) * width
    right_shift = random.uniform(-0.2, 0.2) * width
    top_shift = random.uniform(-0.2, 0.2) * height
    bottom_shift = random.uniform(-0.2, 0.2) * height

    coeffs = (
        left_shift, top_shift,                     # Top-left corner
        width - right_shift, top_shift,            # Top-right corner
        width - right_shift, height - bottom_shift,# Bottom-right corner
        left_shift, height - bottom_shift          # Bottom-left corner
    )
    
    return image.transform((width, height), Image.Transform.QUAD, coeffs, resample=Image.Resampling.BICUBIC)

def add_gaussian_noise(image, std, mean=0):
    np_image = np.array(image)
    noise = np.random.normal(mean, std, np_image.shape).astype(np.int16)
    noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def zoom_image(image, zoom_factor):
    width, height = image.size
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    # Resize the image to zoom in or out
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Crop or pad the image to fit the original dimensions
    left = (new_width - width) / 2
    top = (new_height - height) / 2
    right = (new_width + width) / 2
    bottom = (new_height + height) / 2

    cropped_image = resized_image.crop((left, top, right, bottom))

    return cropped_image

def augment_image(image_path, output_folder, num_variations):
    # Open the original image
    image = Image.open(image_path)

    avg_border_color = get_border_average_color(image)
    avg_border_color_with_alpha = avg_border_color + (255,)
    os.makedirs(output_folder, exist_ok=True)

    for i in range(num_variations):
        # Zoom logic before rotation
        zoom_factor = random.uniform(0.9, 1.9)  # Zoom in or out
        zoomed_image = zoom_image(image, zoom_factor)

        if random.choice([False, True, False, False]):
            zoomed_image = zoomed_image.rotate(180)

        # Random rotation after zoom
        angle = random.randint(-20, 20)
        rotated_image = zoomed_image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        
        if rotated_image.mode != 'RGBA':
            rotated_image = rotated_image.convert('RGBA')
        
        perspective_image = rotated_image # apply_perspective_transform(rotated_image)
        
        background = Image.new('RGBA', perspective_image.size, avg_border_color_with_alpha)
        background.paste(perspective_image, (0, 0), perspective_image)
        
        final_image = background.convert('RGB')
        
        np_image = np.array(final_image)
        black_pixels = np.all(np_image == [0, 0, 0], axis=-1)
        np_image[black_pixels] = avg_border_color
        
        final_image = Image.fromarray(np_image, 'RGB')

        enhancer = ImageEnhance.Brightness(final_image)
        image_bright = enhancer.enhance(random.uniform(0.8, 1.2))

        if random.choice([True, False]):
            image_noised = add_gaussian_noise(image_bright, random.randint(0, 25))
        else:
            image_noised = image_bright
        
        output_image_path = os.path.join(output_folder, f"{os.path.basename(output_folder)}_{i}.jpg")
        image_noised.save(output_image_path)

# Loop through all images in the /patches-input/ directory
input_directory = "static_patches_input"
output_base_directory = "static_patches_input_augmented"

os.makedirs(output_base_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        input_image_path = os.path.join(input_directory, filename)
        output_folder_name = os.path.splitext(filename)[0]
        output_folder_path = os.path.join(output_base_directory, output_folder_name)
        
        augment_image(input_image_path, output_folder_path, num_variations=30)
