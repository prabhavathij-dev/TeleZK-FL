"""
MobileNetV2 (2D) for CheXpert Chest X-Ray Classification.

Uses the pretrained MobileNetV2 from torchvision with a replaced
classifier head for 5-class multi-label classification.
~3.4M parameters, depthwise separable convolutions.

Input:  (batch, 3, 224, 224)
Output: (batch, num_classes) with sigmoid activation
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def get_mobilenetv2_2d(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """Create MobileNetV2 for chest X-ray classification.

    Args:
        num_classes: Number of output classes (pathologies).
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        MobileNetV2 model with modified classifier.
    """
    if pretrained:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = mobilenet_v2(weights=None)

    # Replace classifier for multi-label classification
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes),
        nn.Sigmoid(),
    )

    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
