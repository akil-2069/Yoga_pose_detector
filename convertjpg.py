import os
from PIL import Image

# Path to your dataset images folder
base_dir = "dataset/images"

# List of image extensions to convert
valid_exts = [".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".webp", ".JPG", ".JPEG"]

for root, dirs, files in os.walk(base_dir):
    for file in files:
        name, ext = os.path.splitext(file)
        ext_lower = ext.lower()
        old_path = os.path.join(root, file)

        if ext_lower in valid_exts:
            # Open image and convert to RGB
            img = Image.open(old_path).convert("RGB")
            # Save as .jpg
            new_file = name + ".jpg"
            new_path = os.path.join(root, new_file)
            img.save(new_path, "JPEG")
            
            # Remove old file if not same as new
            if old_path != new_path:
                os.remove(old_path)

print("✅ All images converted to .jpg regardless of original format")
