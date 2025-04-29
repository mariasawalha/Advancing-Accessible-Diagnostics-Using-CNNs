# main.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess_image # type: ignore

input_folder = "fundus_images"
output_folder = "processed_images"
os.makedirs(output_folder, exist_ok=True)

processed_images = []

for file in os.listdir(input_folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            img_path = os.path.join(input_folder, file)
            processed = preprocess_image(img_path)
            processed_images.append(processed)

            # Save to disk (optional)
            save_path = os.path.join(output_folder, file)
            cv2.imwrite(save_path, (processed.squeeze() * 255).astype(np.uint8))
        except Exception as e:
            print(f"Failed to process {file}: {e}")

print(f"Total images processed: {len(processed_images)}")

# Optional: Preview one
if processed_images:
    plt.imshow(processed_images[0].squeeze(), cmap='gray')
    plt.title("Sample Preprocessed Image")
    plt.axis('off')
    plt.show()
