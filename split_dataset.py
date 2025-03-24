import os
import shutil
import random

# Set paths
input_path = r"C:\Users\reach\Desktop\scream_detector\data_spectrogram"
output_path = r"C:\Users\reach\Desktop\scream_detector\split_data"

train_path = os.path.join(output_path, "train")
val_path = os.path.join(output_path, "val")
test_path = os.path.join(output_path, "test")

# Split ratios
train_ratio = 0.7  # 70% training
val_ratio = 0.2    # 20% validation
test_ratio = 0.1   # 10% testing

# Get all class folders
categories = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]

# Create directories dynamically
for category in categories:
    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(val_path, category), exist_ok=True)
    os.makedirs(os.path.join(test_path, category), exist_ok=True)

# Split data for each category
for category in categories:
    category_path = os.path.join(input_path, category)
    files = [f for f in os.listdir(category_path) if f.endswith(".png")]  # Process only spectrogram images
    random.shuffle(files)

    train_split = int(len(files) * train_ratio)
    val_split = train_split + int(len(files) * val_ratio)

    train_files = files[:train_split]
    val_files = files[train_split:val_split]
    test_files = files[val_split:]

    for file in train_files:
        shutil.copy(os.path.join(category_path, file), os.path.join(train_path, category, file))
    
    for file in val_files:
        shutil.copy(os.path.join(category_path, file), os.path.join(val_path, category, file))
    
    for file in test_files:
        shutil.copy(os.path.join(category_path, file), os.path.join(test_path, category, file))

print("Dataset split complete!")
