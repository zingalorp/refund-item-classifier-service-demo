import os
import sys
from typing import Any, Dict, List, Set, Tuple

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import config

# Define the transformations for the datasets
train_transforms: transforms.Compose = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
])

validation_test_transforms: transforms.Compose = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


class FilteredSubset(Subset):
    """
    A custom Subset class that also holds the remapped targets,
    making it compatible with WeightedRandomSampler.
    """

    def __init__(
        self, dataset: Any, indices: List[int], remapped_targets: torch.Tensor
    ):
        super().__init__(dataset, indices)
        self.targets: torch.Tensor = remapped_targets


def get_loaders(
    num_workers: int = config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Creates and returns data loaders. Intelligently filters the dataset
    based on the CLASSES_TO_TRAIN list in the config file.
    """
    # Load full datasets first to get all class information
    full_train_dataset: ImageFolder = ImageFolder(root=config.TRAIN_DIR)
    full_val_dataset: ImageFolder = ImageFolder(root=config.VAL_DIR)

    # Get the mapping from the full dataset (e.g., {'angry': 0, 'disgust': 1, ...})
    full_class_to_idx: Dict[str, int] = full_train_dataset.class_to_idx

    # 1. Identify indices of desired classes
    desired_class_indices: Set[int] = {
        full_class_to_idx[name]
        for name in config.CLASSES_TO_TRAIN
        if name in full_class_to_idx
    }
    if len(desired_class_indices) != len(config.CLASSES_TO_TRAIN):
        print(
            "Warning: Some classes in CLASSES_TO_TRAIN were not found in the dataset directory."
        )

    # 2. Create index masks and remap targets for TRAINING set
    train_indices_to_keep: List[int] = [
        i
        for i, target in enumerate(full_train_dataset.targets)
        if target in desired_class_indices
    ]

    # Create a mapping from old indices (e.g., 0, 2, 4, 5) to new ones (0, 1, 2, 3)
    # Sorting ensures a consistent order, e.g., angry=0, happy=1, sad=2, surprise=3
    sorted_kept_indices: List[int] = sorted(list(desired_class_indices))
    label_remap: Dict[int, int] = {
        old_idx: new_idx for new_idx, old_idx in enumerate(sorted_kept_indices)
    }

    # Get original targets and remap them to the new 0-N range
    train_targets_original: List[int] = [
        full_train_dataset.targets[i] for i in train_indices_to_keep
    ]
    train_targets_remapped: torch.Tensor = torch.tensor(
        [label_remap[t] for t in train_targets_original]
    )

    # 3. Create index masks and remap targets for VALIDATION set
    val_indices_to_keep: List[int] = [
        i
        for i, target in enumerate(full_val_dataset.targets)
        if target in desired_class_indices
    ]
    val_targets_original: List[int] = [
        full_val_dataset.targets[i] for i in val_indices_to_keep
    ]
    val_targets_remapped: torch.Tensor = torch.tensor(
        [label_remap[t] for t in val_targets_original]
    )

    # 4. Create the final Subset datasets with the correct transforms
    # Apply augmentations for the training set
    train_dataset_with_transforms: ImageFolder = ImageFolder(
        root=config.TRAIN_DIR, transform=train_transforms
    )
    train_subset: FilteredSubset = FilteredSubset(
        train_dataset_with_transforms, train_indices_to_keep, train_targets_remapped
    )

    # No augmentations for the validation set
    val_dataset_with_transforms: ImageFolder = ImageFolder(
        root=config.VAL_DIR, transform=validation_test_transforms
    )
    validation_subset: FilteredSubset = FilteredSubset(
        val_dataset_with_transforms, val_indices_to_keep, val_targets_remapped
    )

    print("--- Dataset Loading & Filtering ---")
    print(f"Training {config.VERSION_NAME} on {config.TRAINING_NUM_CLASSES} classes: {config.CLASSES_TO_TRAIN}")
    print(f"Original training set size: {len(full_train_dataset)}")
    print(f"Filtered training set size: {len(train_subset)}")
    print(f"Original validation set size: {len(full_val_dataset)}")
    print(f"Filtered validation set size: {len(validation_subset)}")
    print("-----------------------------------")

    # 5. Handle Class Imbalance for the FILTERED Training Set
    class_counts: torch.Tensor = torch.bincount(train_subset.targets)
    class_weights: torch.Tensor = 1.0 / class_counts.float()
    sample_weights: torch.Tensor = class_weights[train_subset.targets]

    sampler: WeightedRandomSampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # 6. Create DataLoader objects from the Subsets
    train_loader: DataLoader = DataLoader(
        dataset=train_subset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    validation_loader: DataLoader = DataLoader(
        dataset=validation_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # For compatibility, return the test loader and final mapping if needed
    test_dataset: ImageFolder = ImageFolder(
        root=config.TEST_DIR, transform=validation_test_transforms
    )
    test_loader: DataLoader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    final_class_to_idx: Dict[str, int] = {
        name: i for i, name in enumerate(config.CLASSES_TO_TRAIN)
    }

    return train_loader, validation_loader, test_loader, final_class_to_idx


# run 'python src/dataset.py' to test it
if __name__ == "__main__":
    (
        train_loader,
        val_loader,
        test_loader,
        class_mapping,
    ) = get_loaders()

    print(f"\nNumber of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    print(f"Class mapping: {class_mapping}")

    # Check one batch to verify shapes and types
    images, labels = next(iter(train_loader))
    print("\n--- Batch Test ---")
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Label dtype: {labels.dtype}")
    print("--------------------")