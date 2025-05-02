# explainability_scripts.py
import cv2
import numpy as np
import tensorflow as tf
import os
import re

def generate_gradcam(model, image_path, model_name, output_dir="explainability_outputs/gradcam"):
    """
    Generates and saves Grad-CAM heatmap for a given model and input image.
    Supports any image extension (.jpg, .png, etc.).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    input_img = np.expand_dims(img_resized / 255.0, axis=0)

    # Identify last convolutional layer
    conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(conv_layer_name).output, model.output])

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_img)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]

    # Generate heatmap
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap
    superimposed_img = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

    base = os.path.basename(image_path)
    base_no_ext = re.sub(r"\.(jpg|jpeg|png)$", "", base, flags=re.IGNORECASE)
    filename = f"{base_no_ext}_{model_name}_gradcam.jpg"
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, superimposed_img)
    print(f"[GradCAM] Saved: {save_path}")

