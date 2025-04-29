import os
import pandas as pd
import shutil

# Define paths
csv_path = "/Volumes/misc/DBA_final_project/labels.csv"
src_folder = "/Volumes/misc/DBA_final_project/processed_images"
dst_root = "/Volumes/misc/DBA_final_project/sorted_dataset"

# Mapping from code to class name
label_map = {
    "N": "Normal",
    "G": "Glaucoma",
    "A": "AMD",
    "D": "DiabeticRetinopathy"
}

# Read CSV
df = pd.read_csv(csv_path)

# Iterate through each row
for _, row in df.iterrows():
    filename = row["image_name"]
    code = str(row["labels"]).strip().upper()
    class_name = label_map.get(code)

    if class_name is None:
        print(f"Unknown label '{code}' for file '{filename}'")
        continue

    src_path = os.path.join(src_folder, filename)
    dst_folder = os.path.join(dst_root, class_name)
    dst_path = os.path.join(dst_folder, filename)

    os.makedirs(dst_folder, exist_ok=True)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"File not found: {src_path}")

print("Image copying complete")
