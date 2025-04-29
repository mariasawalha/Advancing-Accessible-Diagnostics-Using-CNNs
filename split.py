import os
import shutil
import random

# Paths
source_dir = "/Volumes/misc/DBA_final_project/sorted_dataset"
output_base = "/Volumes/misc/DBA_final_project/split_dataset"
train_dir = os.path.join(output_base, "train")
val_dir = os.path.join(output_base, "val")
test_dir = os.path.join(output_base, "test")

# Parameters
train_split = 0.8
val_split = 0.1
test_split = 0.1

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for label in os.listdir(source_dir):
    label_path = os.path.join(source_dir, label)
    if not os.path.isdir(label_path):
        continue

    images = os.listdir(label_path)
    random.shuffle(images)

    train_end = int(len(images) * train_split)
    val_end = train_end + int(len(images) * val_split)

    split_data = {
        train_dir: images[:train_end],
        val_dir: images[train_end:val_end],
        test_dir: images[val_end:]
    }

    for split_folder, file_list in split_data.items():
        split_label_path = os.path.join(split_folder, label)
        os.makedirs(split_label_path, exist_ok=True)
        for filename in file_list:
            src = os.path.join(label_path, filename)
            dst = os.path.join(split_label_path, filename)
            shutil.copy2(src, dst)

print("Dataset split into train/val/test.")
