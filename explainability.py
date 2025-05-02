import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import shap
import pickle
import os
import h5py

# ------------------- Config -------------------
class_names = ["AMD", "DiabeticRetinopathy", "Glaucoma", "Normal"]
img_size = (224, 224)

# ------------------- Load Data -------------------
X_test_comb = np.load("X_test_comb.npy")
y_test = np.load("y_test.npy")

# For Grad-CAM, we need the original test images
# Let's assume you have X_test_paths.npy (list of file paths)
try:
    with open("X_test_paths.npy", "rb") as f:
        X_test_paths = np.load(f, allow_pickle=True)
except Exception:
    print("Please provide X_test_paths.npy (list of test image file paths) for Grad-CAM.")
    X_test_paths = None

# ------------------- Grad-CAM for CNNs -------------------
def get_img_array(img_path, size):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    outputs = [model.get_layer(last_conv_layer_name).output]
    if isinstance(model.output, list):
        outputs.extend(model.output)
    else:
        outputs.append(model.output)
    grad_model = Model(model.inputs, outputs)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_plot_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(cam_path)
    plt.close()

# Grad-CAM for MobileNetV2 and EfficientNetB0
cnn_models = {
    "MobileNetV2": ("MobileNetV2_model.keras", "Conv_1"),
    "EfficientNetB0": ("EfficientNetB0_model.keras", "top_conv")
}

import tensorflow as tf

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2., alpha=0.25, **kwargs):
        super().__init__(name="focal_loss_1", **kwargs)
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

if X_test_paths is not None:
    for model_name, (model_path, last_conv) in cnn_models.items():
        print(f"\nGrad-CAM for {model_name}")
        # Register the custom loss globally for Keras
        tf.keras.utils.get_custom_objects()['focal_loss_1'] = FocalLoss
        model = load_model(model_path, custom_objects={'focal_loss_1': FocalLoss}, compile=False)
        for i, img_path in enumerate(X_test_paths[:5]):  # Just first 5 images
            img_array = get_img_array(img_path, img_size)
            preds = model.predict(img_array)
            pred_class = np.argmax(preds[0])
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name=last_conv, pred_index=pred_class)
            cam_path = f"gradcam_{model_name}_{i}.png"
            save_and_plot_gradcam(img_path, heatmap, cam_path)
            print(f"Saved Grad-CAM for {img_path} as {cam_path}")
else:
    print("Skipping Grad-CAM: X_test_paths not found.")

# ------------------- SHAP for XGBoost -------------------
with open("cat_model.pkl", "rb") as f:
    ensemble = pickle.load(f)

# Get XGBoost and RandomForest from VotingClassifier
xgb = ensemble.named_estimators_["xgb"]
rf = ensemble.named_estimators_["rf"]

print("\nExplaining XGBoost with SHAP...")
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test_comb)
shap.summary_plot(shap_values, X_test_comb, show=False)
plt.title("SHAP Summary Plot (XGBoost)")
plt.tight_layout()
plt.savefig("shap_xgb_summary.png")
plt.close()

# ------------------- Feature Importance for RandomForest -------------------
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 4))
plt.title("RandomForest Feature Importances")
plt.bar(range(20), importances[indices][:20], align="center")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
plt.close()

print("\nExplainability results saved:")
print("- Grad-CAM images for CNNs (if X_test_paths.npy provided)")
print("- SHAP summary plot for XGBoost: shap_xgb_summary.png")
print("- RandomForest feature importance: rf_feature_importance.png")
