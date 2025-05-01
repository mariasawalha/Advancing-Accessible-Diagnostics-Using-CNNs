import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, VerticalFlip, Rotate, ElasticTransform, GridDistortion, RandomResizedCrop, Resize
from albumentations.core.composition import OneOf

# ----------------------- Config -----------------------
data_dir = "/Volumes/misc/DBA_final_project/split_dataset"
img_size = (224, 224)
batch_size = 32
epochs_transfer = 5
epochs_finetune = 3
class_names = ["AMD", "DiabeticRetinopathy", "Glaucoma", "Normal"]

# --------------------- Augmentation --------------------
albu_transform = Compose([
    Resize(224, 224),
    RandomBrightnessContrast(0.1, 0.1, p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    Rotate(limit=15, p=0.5),
    OneOf([
        ElasticTransform(alpha=1.0, sigma=50, p=0.5),  
        GridDistortion(p=0.5)
    ], p=0.3)
    # Note: Normalize() is intentionally removed
])

def albu_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = albu_transform(image=image)['image']
    return image

# --------------------- Data Generator -------------------
from tensorflow.keras.utils import Sequence
class AlbuDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        valid_indices = []
        for i, file_path in enumerate(batch_x):
            try:
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Warning: Could not read image at {file_path}")
                    continue
                if self.augment:
                    img = albu_preprocess(img)
                else:
                    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), img_size)
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error processing image {file_path}: {str(e)}")
                continue
        
        if not images:
            raise ValueError(f"No valid images found in batch starting at index {idx}")
        
        batch_y = [batch_y[i] for i in valid_indices]
        return np.array(images), tf.keras.utils.to_categorical(batch_y, num_classes=len(class_names))

# --------------------- Focal Loss ---------------------
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))
    return loss

# ------------------ Training Pipeline ------------------
def build_light_cnn(model_fn, name):
    base = model_fn(include_top=False, weights='imagenet', input_shape=img_size + (3,))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(len(class_names), activation='softmax')(x)
    model = Model(base.input, output)
    model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs_transfer,
              callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])

    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False
    model.compile(optimizer=Adam(1e-5), loss=focal_loss(), metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs_finetune,
              callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])
    return model

def extract_features(model, generator):
    extractor = Model(inputs=model.input, outputs=model.layers[-3].output)
    features = extractor.predict(generator, verbose=1)
    return features

# ------------------ Load Data ------------------
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

image_paths = []
for class_name in class_names:
    class_paths = glob(f"{data_dir}/*/{class_name}/*.jpg") + glob(f"{data_dir}/*/{class_name}/*.png") + glob(f"{data_dir}/*/{class_name}/*.jpeg")
    image_paths.extend(class_paths)

if not image_paths:
    raise ValueError(f"No images found in {data_dir}. Please check the directory structure and file extensions.")

labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
label_enc = LabelEncoder()
y = label_enc.fit_transform(labels)

print(f"Found {len(image_paths)} images in total")
for class_name in class_names:
    class_count = sum(1 for label in labels if label == class_name)
    print(f"{class_name}: {class_count} images")

X_trainval, X_test, y_trainval, y_test = train_test_split(image_paths, y, test_size=0.2, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval)

print(f"\nSplit sizes:")
print(f"Training: {len(X_train)} images")
print(f"Validation: {len(X_val)} images")
print(f"Test: {len(X_test)} images")

train_gen = AlbuDataGenerator(X_train, y_train, batch_size, augment=True)
val_gen = AlbuDataGenerator(X_val, y_val, batch_size, augment=False)
test_gen = AlbuDataGenerator(X_test, y_test, batch_size, augment=False)

# ------------------ Train CNNs & Extract ------------------
models = {
    "MobileNetV2": MobileNetV2,
    "EfficientNetB0": EfficientNetB0
}
X_train_list, X_val_list, X_test_list = [], [], []

for name, fn in models.items():
    model = build_light_cnn(fn, name)
    X_train_list.append(extract_features(model, train_gen))
    X_val_list.append(extract_features(model, val_gen))
    X_test_list.append(extract_features(model, test_gen))

X_train_comb = np.concatenate(X_train_list, axis=1)
X_val_comb = np.concatenate(X_val_list, axis=1)
X_test_comb = np.concatenate(X_test_list, axis=1)

# ------------------ Meta-Classifier Ensemble ------------------
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05)
rf = RandomForestClassifier(n_estimators=800)
ensemble = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf)], voting='soft')
ensemble.fit(X_train_comb, y_train)

# ------------------ Evaluation ------------------
y_pred = ensemble.predict(X_test_comb)
y_probs = ensemble.predict_proba(X_test_comb)
y_test_bin = label_binarize(y_test, classes=range(len(class_names)))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

auc_scores = roc_auc_score(y_test_bin, y_probs, average=None)
for i, score in enumerate(auc_scores):
    print(f"{class_names[i]} AUC: {score:.4f}")
