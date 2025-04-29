import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Important for consistency

    # Apply CLAHE to each channel separately
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(img)
    clahe_channels = [clahe.apply(c) for c in channels]
    enhanced_img = cv2.merge(clahe_channels)

    normalized_img = enhanced_img.astype(np.float32) / 255.0

    return normalized_img  # Shape (224,224,3)
