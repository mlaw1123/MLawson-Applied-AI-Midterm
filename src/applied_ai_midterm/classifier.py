"""MobileNetV2 binary classifier used by Model A and later Model B."""

from __future__ import annotations

from torch import nn
from torchvision.models import MobileNet_V2_Weights, MobileNetV2, mobilenet_v2


def create_binary_mobilenet(*, pretrained: bool = True) -> MobileNetV2:
    """Create MobileNetV2 with one output logit.

    Set ``pretrained=False`` in tests and offline smoke checks to guarantee that
    torchvision does not attempt a network download.
    """
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, 1)
    return model
