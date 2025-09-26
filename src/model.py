import torch
import torch.nn as nn
from typing import List, Type


# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Re-calibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Use 1D convs instead of Linear layers for efficiency
        self.fc = nn.Sequential(
            nn.Conv1d(
                channels, channels // reduction, kernel_size=1, padding=0, bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                channels // reduction, channels, kernel_size=1, padding=0, bias=False
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the SE block."""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# Modified Residual SE Block
class ResidualSEBlock(nn.Module):
    """
    A residual block with a Squeeze-and-Excitation block integrated.
    Includes a shortcut connection to handle changes in dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection to match dimensions if they change (due to stride or channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Residual SE block."""
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


# The Classification Model
class ClassificationResNetSE(nn.Module):
    """
    A ResNet-style architecture with integrated SE blocks for image classification.
    Designed for efficient classification on small images.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        num_blocks: List[int] = [2, 2, 2, 2],
        num_filters: List[int] = [64, 128, 256, 512],
    ) -> None:
        super().__init__()

        self.in_planes: int = num_filters[0]

        # Initial convolution layer
        self.initial_conv = nn.Conv2d(
            in_channels, self.in_planes, kernel_size=3, padding=1, bias=False
        )
        self.initial_bn = nn.BatchNorm2d(self.in_planes)
        self.initial_relu = nn.ReLU(inplace=True)

        # Create layers of residual blocks
        self.layer1 = self._make_layer(
            ResidualSEBlock, num_filters[0], num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            ResidualSEBlock, num_filters[1], num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            ResidualSEBlock, num_filters[2], num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            ResidualSEBlock, num_filters[3], num_blocks[3], stride=2
        )

        # Final layers for classification
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_filters[3], num_classes)

    def _make_layer(
        self, block: Type[ResidualSEBlock], planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Helper function to create a layer of residual blocks."""
        strides: List[int] = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the full ClassificationResNetSE model."""
        out = self.initial_relu(self.initial_bn(self.initial_conv(x)))
        out = self.layer1(out)  # 64 filters, 48x48
        out = self.layer2(out)  # 128 filters, 24x24
        out = self.layer3(out)  # 256 filters, 12x12
        out = self.layer4(out)  # 512 filters, 6x6

        out = self.avg_pool(out)  # (batch, 512, 1, 1)
        out = out.view(out.size(0), -1)  # Flatten -> (batch, 512)
        out = self.classifier(out)

        return out


# For compatibility
ClassificationCNN = ClassificationResNetSE

if __name__ == "__main__":
    # Test the model with a dummy input
    # Use default settings for a simple test
    model: ClassificationResNetSE = ClassificationResNetSE(num_classes=6)
    dummy_input: torch.Tensor = torch.randn(64, 1, 48, 48)
    output: torch.Tensor = model(dummy_input)
    print(f"Model output shape: {output.shape}")  # Should be [64, 6]

    # Count parameters to see model complexity
    total_params: int = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total trainable parameters: {total_params:,}")
