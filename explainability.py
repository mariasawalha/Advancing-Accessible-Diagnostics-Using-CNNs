# -------------------- Explainability Runner Script --------------------
import os
import numpy as np
import cv2
import shap
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from explainability_scripts import generate_gradcam
from sklearn.preprocessing import label_binarize
import pickle

# -------------------- Output Folder --------------------
os.makedirs("explainability_outputs/gradcam", exist_ok=True)
os.makedirs("explainability_outputs/shap", exist_ok=True)

# -------------------- Load CNN Models --------------------
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2., alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha
        })
        return config

class_names = ["AMD", "DiabeticRetinopathy", "Glaucoma", "Normal"]

print("Starting to load models...")
try:
    mobilenet_model = load_model("MobileNetV2_model.h5", custom_objects={"FocalLoss": FocalLoss})
    print("Successfully loaded MobileNetV2 model")
except Exception as e:
    print(f"Error loading MobileNetV2 model: {str(e)}")
    raise

try:
    efficientnet_model = load_model("EfficientNetB0_model.h5", custom_objects={"FocalLoss": FocalLoss})
    print("Successfully loaded EfficientNetB0 model")
except Exception as e:
    print(f"Error loading EfficientNetB0 model: {str(e)}")
    raise

# -------------------- Load Ensemble Data --------------------
X_test_comb = np.load("X_test_comb.npy")
y_test = np.load("y_test.npy")

with open("cat_model.pkl", "rb") as f:
    cat = pickle.load(f)

# -------------------- Modified Grad-CAM to Save Plots --------------------
def save_gradcam(model, image_path, model_name):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    input_img = np.expand_dims(img_resized / 255.0, axis=0)

    layer_name = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_img)
        class_index = np.argmax(predictions[0])
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    filename = os.path.basename(image_path).replace(".jpg", f"_{model_name}_gradcam.jpg")
    save_path = os.path.join("explainability_outputs/gradcam", filename)
    cv2.imwrite(save_path, superimposed_img)
    print(f"Saved Grad-CAM: {save_path}")

# -------------------- Sample Grad-CAM Visualizations --------------------
sample_image_paths = [
    "/Volumes/misc/DBA_final_project/split_dataset/test/AMD/0_RFiMD_30_ARMD.png",
    "/Volumes/misc/DBA_final_project/split_dataset/test/DiabeticRetinopathy/8bed09514c3b.png",
    "/Volumes/misc/DBA_final_project/split_dataset/test/Glaucoma/EyePACS-DEV-RG-341.jpg"
]

for img_path in sample_image_paths:
    print(f"\nGrad-CAM for: {img_path}")
    save_gradcam(mobilenet_model, img_path, "MobileNetV2")
    save_gradcam(efficientnet_model, img_path, "EfficientNetB0")

# -------------------- SHAP for Ensemble Classifier --------------------
explainer = shap.Explainer(cat)
shap_values = explainer(X_test_comb[:100])

shap.summary_plot(
    shap_values,
    pd.DataFrame(X_test_comb[:100]),
    feature_names=[f"f{i}" for i in range(X_test_comb.shape[1])],
    show=False
)
plt.tight_layout()
shap_path = "explainability_outputs/shap/shap_summary_plot.png"
plt.savefig(shap_path)
plt.close()
print(f"Saved SHAP summary plot to: {shap_path}")
