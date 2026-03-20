"""
MobileNetV2 (1D) for PTB-XL ECG Classification.

1D adaptation of MobileNetV2 for 12-lead ECG time-series classification.
All 2D operations are converted to 1D equivalents:
  Conv2d -> Conv1d, BatchNorm2d -> BatchNorm1d, AdaptiveAvgPool2d -> AdaptiveAvgPool1d

Architecture preserves the full MobileNetV2 inverted residual structure.
~2.1M parameters.

Input:  (batch, 12, 1000) — 12 ECG leads, 1000 time steps at 100Hz
Output: (batch, num_classes) with sigmoid activation
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class ConvBNReLU1d(nn.Sequential):
    """Conv1d + BatchNorm1d + ReLU6 block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual1d(nn.Module):
    """1D inverted residual block (MobileNetV2 building block).

    Structure: (expand 1x1) -> (depthwise conv1d) -> (project 1x1)
    Residual connection when stride=1 and in_channels == out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 1,
    ):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []

        # Expand phase (1x1 conv) — skip if expand_ratio == 1
        if expand_ratio != 1:
            layers.append(ConvBNReLU1d(in_channels, hidden_dim, kernel_size=1))

        # Depthwise convolution
        layers.append(
            ConvBNReLU1d(hidden_dim, hidden_dim, kernel_size=3,
                        stride=stride, groups=hidden_dim)
        )

        # Pointwise (project) — linear bottleneck, no activation
        layers.extend([
            nn.Conv1d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2_1D(nn.Module):
    """1D MobileNetV2 for ECG classification.

    Inverted residual block configuration (same as original MobileNetV2):
    [expand_ratio, output_channels, num_blocks, stride]
    """

    # Standard MobileNetV2 block specs
    BLOCK_CONFIGS: List[Tuple[int, int, int, int]] = [
        # t, c, n, s
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        width_mult: float = 1.0,
    ):
        super().__init__()

        input_channel = int(32 * width_mult)
        last_channel = int(1280 * max(1.0, width_mult))

        # Initial convolution layer
        features = [ConvBNReLU1d(in_channels, input_channel, kernel_size=3, stride=2)]

        # Build inverted residual blocks
        for t, c, n, s in self.BLOCK_CONFIGS:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual1d(input_channel, output_channel,
                                      stride=stride, expand_ratio=t)
                )
                input_channel = output_channel

        # Final 1x1 convolution to 1280 channels
        features.append(ConvBNReLU1d(input_channel, last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
            nn.Sigmoid(),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using standard practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_mobilenetv2_1d(
    in_channels: int = 12,
    num_classes: int = 5,
    pretrained: bool = False,
) -> nn.Module:
    """Create 1D MobileNetV2 for ECG classification.

    Args:
        in_channels: Number of ECG leads (typically 12).
        num_classes: Number of output classes.
        pretrained: Unused — no pretrained weights for 1D variant.

    Returns:
        MobileNetV2_1D model.
    """
    # pretrained flag is accepted for API consistency but not used
    model = MobileNetV2_1D(
        in_channels=in_channels,
        num_classes=num_classes,
    )
    return model
