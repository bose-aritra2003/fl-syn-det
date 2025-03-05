import os
import cv2
import numpy as np
# from models.EfficientNetB0Pretrained import EfficientNetB0Pretrained
from models.SimpleCNN import SimpleCNN

# Define class labels
class_labels = {"real": 0, "fake": 1}

def load_dataset(dataset_path):
    """Loads dataset from the given path.
    
    Args:
        dataset_path (str): Path to dataset (for a client or server).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Images and corresponding labels.
    """
    images, labels = [], []
    print(f"Loading dataset from: {dataset_path}")

    for folder in class_labels:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            continue  # Skip if folder does not exist

        label = class_labels[folder]

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            image = cv2.imread(img_path)

            if image is None:
                continue  # Skip corrupted images

            # Normalize pixel values (0 to 1)
            image = image.astype("float32") / 255.0

            images.append(image)
            labels.append(label)

    images = np.array(images, dtype="float32")
    labels = np.array(labels, dtype="int32")  # No one-hot encoding

    print(f"Dataset loaded: {images.shape[0]} images")
    print(f"Image shape: {images.shape}")

    return images, labels


def get_model():
    model = SimpleCNN()
    return model