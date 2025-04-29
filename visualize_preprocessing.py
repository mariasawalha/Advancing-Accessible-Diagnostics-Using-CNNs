import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import random
import os

# ------------------- Strong Preprocessing -------------------
def preprocess_strong(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(image)
    clahe_channels = [clahe.apply(c) for c in channels]
    enhanced_img = cv2.merge(clahe_channels)
    return enhanced_img.astype(np.float32) / 255.0

# ------------------- Gentle Preprocessing -------------------
gentle_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=1.0),
    A.Rotate(limit=15, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
    A.Normalize()
])
def preprocess_gentle(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return gentle_transform(image=image)['image']

# ------------------- Visualization & Export -------------------
def export_comparison_grid(folder_path, num_samples=5, output_path='preprocessing_comparison.png'):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
    sampled_paths = random.sample(image_files, min(num_samples, len(image_files)))

    fig, axes = plt.subplots(len(sampled_paths), 3, figsize=(12, 4 * len(sampled_paths)))
    if len(sampled_paths) == 1:
        axes = [axes]  # handle single row

    for i, path in enumerate(sampled_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        original = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB) / 255.0
        strong = preprocess_strong(img)
        gentle = preprocess_gentle(img)

        images = [original, strong, gentle]
        titles = ["Original RGB", "Strong (CLAHE + Aug)", "Gentle (Safe Aug)"]

        for j in range(3):
            axes[i][j].imshow(images[j])
            axes[i][j].axis('off')
            if i == 0:
                axes[i][j].set_title(titles[j], fontsize=14)

    # Add caption below the full figure
    plt.figtext(0.5, 0.02, 
        "Figure: Visual comparison of original retinal images vs. two preprocessing pipelines. "
        "The strong preprocessing pipeline includes CLAHE and geometric augmentations, while the gentle approach preserves retinal anatomy with minimal distortions.",
        wrap=True, horizontalalignment='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Figure saved to: {output_path}")

# ------------------- Run It -------------------
image_folder = "/Volumes/misc/DBA_final_project/processed_images"
export_comparison_grid(image_folder, num_samples=5, output_path='preprocessing_comparison.png')
