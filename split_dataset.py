import os
import shutil
import random

# Paths
DATASET_PATH = "fl-syn-det-ds"
OUTPUT_PATH = "fl-syn-det-ds-split"

# Clients and server folder structure
CLIENTS = ["client_1", "client_2"]
SERVER = "server"

# Ensure output directories exist
for client in CLIENTS:
    os.makedirs(os.path.join(OUTPUT_PATH, client, "train", "0"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, client, "train", "1"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, client, "test", "0"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, client, "test", "1"), exist_ok=True)

os.makedirs(os.path.join(OUTPUT_PATH, SERVER, "test", "0"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, SERVER, "test", "1"), exist_ok=True)

# Function to split and copy files with proper shuffling
def split_and_copy(source_folder, dest_folders, split_ratios):
    """
    Splits files from source_folder into multiple destination folders based on split_ratios.
    Ensures shuffling before distribution.
    """
    for class_label in ["0", "1"]:  # Real (0) and Fake (1)
        class_path = os.path.join(source_folder, class_label)
        files = os.listdir(class_path)

        # Shuffle the dataset for randomness
        random.shuffle(files)

        # Compute split points
        split_points = [int(len(files) * sum(split_ratios[:i])) for i in range(len(split_ratios) + 1)]

        # Copy files to each destination folder
        for i, dest_folder in enumerate(dest_folders):
            dest_class_path = os.path.join(dest_folder, class_label)
            os.makedirs(dest_class_path, exist_ok=True)

            for file in files[split_points[i] : split_points[i + 1]]:
                shutil.copy(os.path.join(class_path, file), os.path.join(dest_class_path, file))

        # Debug: Print some shuffled filenames to verify randomness
        print(f"📌 Sample {class_label} images for verification: {files[:5]}")  # Print first 5 file names

# Split train dataset (50% for each client)
train_source = os.path.join(DATASET_PATH, "train")
train_destinations = [os.path.join(OUTPUT_PATH, client, "train") for client in CLIENTS]
split_and_copy(train_source, train_destinations, [0.5, 0.5])  # 50% each

# Split test dataset (50% for each client, 100% for server)
test_source = os.path.join(DATASET_PATH, "test")
test_destinations = [os.path.join(OUTPUT_PATH, client, "test") for client in CLIENTS]
split_and_copy(test_source, test_destinations, [0.5, 0.5])  # 50% each
split_and_copy(test_source, [os.path.join(OUTPUT_PATH, SERVER, "test")], [1.0])  # 100% to server

print("✅ Dataset successfully shuffled and split for federated learning!")