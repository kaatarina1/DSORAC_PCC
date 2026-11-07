from PIL import Image
from pathlib import Path
import numpy as np

# Images are in data/images (paths were relative to data/ before this script moved to scripts/)
images = Path(__file__).resolve().parents[1] / "data" / "images"

image_path = images / "slika-3_Popravljena.png"
image = Image.open(image_path).convert("RGBA")
resized_image = image.resize((1024, 512), Image.LANCZOS)

def apply_random_alpha(img, alpha_fraction):
    img_array = np.array(img)
    mask = np.random.rand(*img_array.shape[:2]) > alpha_fraction
    img_array[..., 3] = np.where(mask, img_array[..., 3], 0)
    return Image.fromarray(img_array)

# Generate the images
image_50 = apply_random_alpha(resized_image, 0.5)
image_80 = apply_random_alpha(resized_image, 0.8)
image_90 = apply_random_alpha(resized_image, 0.9)
image_95 = apply_random_alpha(resized_image, 0.95)
image_99 = apply_random_alpha(resized_image, 0.99)

# Save the images
paths = [
    images / "image4.png",
    images / "image4_50_alpha.png",
    images / "image4_80_alpha.png",
    images / "image4_90_alpha.png",
    images / "image4_95_alpha.png",
    images / "image4_99_alpha.png",
]

resized_image.save(paths[0])
image_50.save(paths[1])
image_80.save(paths[2])
image_90.save(paths[3])
image_95.save(paths[4])
image_99.save(paths[5])
