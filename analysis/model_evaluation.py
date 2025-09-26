import sys
import os
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from tqdm import tqdm

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import ClassificationCNN  # noqa: E402
from src import config  # noqa: E402


MODEL_PATHS: Dict[str, str] = {
    "fashion_mnist": os.path.join(project_root, "models", "fashion_mnist_v1.pth")
}

# Path to save evaluation figures
FIGURES_SAVE_PATH: str = os.path.join(project_root, "Figures")

TEST_DATA_PATH: str = config.TEST_DIR

# Fashion-MNIST class names (ordered alphabetically as loaded by ImageFolder)
FASHION_MNIST_CLASSES: List[str] = [
    "Ankle_boot", "Bag", "Coat", "Dress", "Pullover",
    "Sandal", "Shirt", "Sneaker", "T-shirt_top", "Trouser"
]

CLASS_NAMES_MAP: Dict[str, List[str]] = {
    "fashion_mnist": FASHION_MNIST_CLASSES
}
NUM_CLASSES_MAP: Dict[str, int] = {"fashion_mnist": 10}


def get_test_loader(test_dir: str) -> Tuple[DataLoader, Dict[str, int]]:
    """Loads the test dataset and returns a DataLoader and class-to-index mapping."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = ImageFolder(root=test_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    return loader, dataset.class_to_idx


def get_predictions_and_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gets predictions, labels, and features from the model."""
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    feature_extractor.eval()

    all_features, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Getting Predictions & Features"):
            inputs = inputs.to(device)
            features = feature_extractor(inputs).view(inputs.size(0), -1)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_features.append(features.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_features)
    )


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], save_path: str = None
) -> None:
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        confusion_matrix_path = os.path.join(
            save_path, "confusion_matrix_fashion_mnist.png")
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {confusion_matrix_path}")


def visualize_features_tsne(
    features: np.ndarray, labels: np.ndarray, class_names: List[str], save_path: str = None
) -> None:
    """Performs t-SNE and visualizes the feature space."""
    print("\nPerforming t-SNE... (this may take a moment)")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=420)
    tsne_results = tsne.fit_transform(features)
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=[class_names[l] for l in labels],
        palette=sns.color_palette("hsv", len(class_names)),
        legend="full",
        alpha=0.7
    )
    plt.title('t-SNE Visualization of Learned Feature Space')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Move legend outside the plot area
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        tsne_path = os.path.join(
            save_path, "tsne_visualization_fashion_mnist.png")
        plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved to: {tsne_path}")


def main() -> None:
    """Main function to run the model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Fashion-MNIST classification model."
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='fashion_mnist',
        choices=['fashion_mnist'],
        help="Specify model to evaluate."
    )
    args = parser.parse_args()

    model_type: str = args.model_type
    print(f"\n--- Evaluating {model_type.replace('_', ' ')} model ---")
    model_path: str = MODEL_PATHS[model_type]
    num_classes: int = NUM_CLASSES_MAP[model_type]
    class_names_for_model: List[str] = CLASS_NAMES_MAP[model_type]

    device: torch.device = torch.device(config.DEVICE)
    model: nn.Module = ClassificationCNN(num_classes=num_classes).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}.")
        return
    print(f"Model loaded from {model_path}")

    try:
        test_loader, class_to_idx = get_test_loader(TEST_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Test data not found at {TEST_DATA_PATH}.")
        return
    print(f"Test data loaded from {TEST_DATA_PATH}")

    predictions, true_labels, features = get_predictions_and_features(
        model, test_loader, device
    )

    # No filtering needed for Fashion-MNIST - use all classes
    predictions_final = predictions
    true_labels_final = true_labels
    features_final = features

    # Display Results
    print("\n--- Classification Report ---")
    report: str = classification_report(
        true_labels_final,
        predictions_final,
        target_names=class_names_for_model
    )
    print(report)

    print("\nDisplaying Confusion Matrix...")
    plot_confusion_matrix(
        true_labels_final, predictions_final, class_names_for_model, FIGURES_SAVE_PATH
    )

    visualize_features_tsne(
        features_final, true_labels_final, class_names_for_model, FIGURES_SAVE_PATH
    )


if __name__ == '__main__':
    main()
