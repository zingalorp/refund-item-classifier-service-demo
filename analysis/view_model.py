import os
import sys

import torch
from torch.utils.tensorboard import SummaryWriter

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import config # noqa: E402
from src.model import ClassificationCNN  # noqa: E402


def main() -> None:
    """
    Writes the model graph to a TensorBoard log file for visualization.
    """
    print("Preparing to write model graph to TensorBoard...")

    device: torch.device = torch.device(config.DEVICE)
    model: ClassificationCNN = ClassificationCNN(
        num_classes=config.TRAINING_NUM_CLASSES).to(device)

    # 1. Create a SummaryWriter
    writer: SummaryWriter = SummaryWriter(
        "runs/fashion_mnist_model_architecture")

    # 2. Get a sample input batch
    # (batch_size, channels, height, width)
    dummy_input: torch.Tensor = torch.randn(
        1, 1, config.IMG_HEIGHT, config.IMG_WIDTH
    ).to(device)

    # 3. Add the graph to the writer
    writer.add_graph(model, dummy_input)

    # 4. Close the writer
    writer.close()

    print("\nModel graph has been successfully written to the 'runs' directory.")
    print("To view the graph, run the following command in your terminal:")
    print("\n  tensorboard --logdir=runs\n")
    print(
        "Then open the URL it provides in your web browser "
        "(usually http://localhost:6006/)."
    )


if __name__ == "__main__":
    main()
