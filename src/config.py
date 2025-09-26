import os
import sys
from typing import List

import torch

current_script_dir: str = os.path.dirname(os.path.abspath(__file__))
project_root: str = os.path.abspath(os.path.join(current_script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Dataset paths
TRAIN_DIR: str = os.path.join(project_root, "data", "train")
VAL_DIR: str = os.path.join(project_root, "data", "test")
TEST_DIR: str = os.path.join(project_root, "data", "test")

# Use CUDA if available, otherwise use CPU
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Image dimensions - Fashion-MNIST native resolution
IMG_HEIGHT: int = 28
IMG_WIDTH: int = 28

# Training hyperparameters
BATCH_SIZE: int = 256
LABEL_SMOOTHING: float = 0.10
WEIGHT_DECAY: float = 0.05
NUM_EPOCHS: int = 70
NUM_WORKERS: int = 2  # Number of CPU cores to use for data loading
EARLY_STOPPING_PATIENCE: int = (
    10  # Number of epochs with no improvement before stopping
)
MAX_LR: float = 0.01

# Fashion-MNIST classes
CLASSES_TO_TRAIN: List[str] = ["T-shirt_top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"]
# Alternative: Use subset of classes for faster training
# CLASSES_TO_TRAIN: List[str] = ["T-shirt_top", "Trouser", "Dress", "Sneaker"]

TRAINING_NUM_CLASSES: int = len(CLASSES_TO_TRAIN)

# Path to save the trained model
VERSION_NAME: str = "fashion_mnist_v1"
MODEL_SAVE_PATH: str = os.path.join(project_root, "models", f"{VERSION_NAME}.pth")
CHECKPOINT_SAVE_PATH: str = os.path.join(
    project_root, "models", "checkpoint.pth.tar"
)

# Plot save paths
PLOT_SAVE_PATH_LOSS: str = os.path.join(
    project_root, "Figures", "plots", f"loss_curve_{VERSION_NAME}.png"
)
PLOT_SAVE_PATH_ACCURACY: str = os.path.join(
    project_root, "Figures", "plots", f"accuracy_curve_{VERSION_NAME}.png"
)