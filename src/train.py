from src.model import ClassificationCNN
from src import config, dataset
import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def save_plots(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
) -> None:
    """Saves loss and accuracy plots to files."""
    plot_dir: str = os.path.dirname(config.PLOT_SAVE_PATH_LOSS)
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(config.PLOT_SAVE_PATH_LOSS)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(config.PLOT_SAVE_PATH_ACCURACY)
    plt.close()


def save_checkpoint(state: Dict[str, Any], filename: str = config.CHECKPOINT_SAVE_PATH) -> None:
    """Saves the training checkpoint."""
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    filename: str = config.CHECKPOINT_SAVE_PATH,
) -> Tuple[int, float, int, List[float], List[float], List[float], List[float]]:
    """Loads a training checkpoint."""
    print("=> Loading checkpoint")
    checkpoint: Dict[str, Any] = torch.load(
        filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        checkpoint["epoch"],
        checkpoint["best_val_accuracy"],
        checkpoint["epochs_no_improve"],
        checkpoint["train_loss_history"],
        checkpoint["val_loss_history"],
        checkpoint["train_acc_history"],
        checkpoint["val_acc_history"],
    )


def train_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """
    Performs one training pass and calculates loss AND accuracy simultaneously.
    Returns the average loss and the accuracy for the epoch.
    """
    model.train()
    loop: tqdm = tqdm(loader, leave=True)
    running_loss: float = 0.0
    num_correct: int = 0
    num_samples: int = 0

    for data, targets in loop:
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        scores: torch.Tensor = model(data)

        # Loss Calculation & Backward Pass
        loss: torch.Tensor = loss_fn(scores, targets)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy Calculation (using the same 'scores' from the forward pass)
        _, predictions = scores.max(1)
        num_correct += (predictions == targets).sum().item()
        num_samples += predictions.size(0)

        loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    avg_loss: float = running_loss / len(loader)
    accuracy: float = (num_correct / num_samples) * 100
    return avg_loss, accuracy


def validate_one_epoch(
    loader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: str
) -> Tuple[float, float]:
    """
    Performs one validation pass, calculating loss and accuracy.
    Returns the average loss and the accuracy for the epoch.
    """
    model.eval()
    running_loss: float = 0.0
    num_correct: int = 0
    num_samples: int = 0

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            scores: torch.Tensor = model(data)

            # Loss Calculation
            loss: torch.Tensor = loss_fn(scores, targets)
            running_loss += loss.item()

            # Accuracy Calculation
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum().item()
            num_samples += predictions.size(0)

    avg_loss: float = running_loss / len(loader)
    accuracy: float = (num_correct / num_samples) * 100
    return avg_loss, accuracy


def main(args: argparse.Namespace) -> None:
    """Main training and validation loop."""
    print(f"Using device: {config.DEVICE}")

    train_loader, val_loader, _, _ = dataset.get_loaders(num_workers=4)

    model: nn.Module = ClassificationCNN(num_classes=config.TRAINING_NUM_CLASSES).to(
        config.DEVICE
    )
    loss_fn: nn.Module = nn.CrossEntropyLoss(
        label_smoothing=config.LABEL_SMOOTHING
    )
    optimizer: optim.Optimizer = optim.AdamW(
        model.parameters(), lr=config.MAX_LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler: optim.lr_scheduler._LRScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
    )

    start_epoch: int = 0
    best_val_accuracy: float = 0.0
    epochs_no_improve: int = 0
    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    train_acc_history: List[float] = []
    val_acc_history: List[float] = []

    if args.resume and os.path.exists(config.CHECKPOINT_SAVE_PATH):
        try:
            (
                start_epoch,
                best_val_accuracy,
                epochs_no_improve,
                train_loss_history,
                val_loss_history,
                train_acc_history,
                val_acc_history,
            ) = load_checkpoint(model, optimizer, scheduler)
            start_epoch += 1
            print(
                f"Resumed from epoch {start_epoch}. Best accuracy: {best_val_accuracy:.2f}%"
            )
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
            start_epoch = 0
    else:
        print("Starting fresh training run.")

    # Training Loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")

        # One pass for training loss and accuracy
        train_loss, train_accuracy = train_one_epoch(
            train_loader, model, optimizer, loss_fn, config.DEVICE
        )
        train_loss_history.append(train_loss)
        train_acc_history.append(train_accuracy)

        # One pass for validation loss and accuracy
        val_loss, val_accuracy = validate_one_epoch(
            val_loader, model, loss_fn, config.DEVICE
        )
        val_loss_history.append(val_loss)
        val_acc_history.append(val_accuracy)

        print(
            f"Training Loss: {train_loss:.4f} | Training Accuracy: {train_accuracy:.2f}%"
        )
        print(
            f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%"
        )

        scheduler.step()
        print(
            f"Learning Rate for next epoch: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Saving and Early Stopping Logic
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            print(
                f"New best validation accuracy: {best_val_accuracy:.2f}%. Saving best model..."
            )
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        checkpoint_state: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "epochs_no_improve": epochs_no_improve,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "train_acc_history": train_acc_history,
            "val_acc_history": val_acc_history,
        }
        save_checkpoint(checkpoint_state)

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered after {epochs_no_improve} epochs.")
            break

    print("\nTraining finished.")
    print(f"Best validation accuracy achieved: {best_val_accuracy:.2f}%")

    print("Saving performance plots...")
    save_plots(
        train_loss_history, val_loss_history, train_acc_history, val_acc_history
    )
    print("Plots saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image Classification Training Script")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint",
    )
    args = parser.parse_args()
    main(args)
