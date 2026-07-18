"""SRGAN generator, discriminator, and configurable generator objective."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import VGG19_Weights, vgg19


class ResidualBlock(nn.Module):
    """Residual SRGAN block that preserves channels and spatial dimensions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Add the learned residual to the input tensor."""
        return inputs + self.block(inputs)


class UpsampleBlock(nn.Module):
    """Double spatial resolution using convolution and pixel shuffle."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return a feature map with twice the input height and width."""
        return self.block(inputs)


class Generator(nn.Module):
    """Generate 128×128 RGB images from 32×32 inputs in range ``[-1, 1]``."""

    def __init__(self, residual_blocks: int = 16, channels: int = 64) -> None:
        super().__init__()
        if residual_blocks <= 0 or channels <= 0:
            raise ValueError("residual_blocks and channels must be greater than zero")
        self.initial = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=9, padding=4),
            nn.PReLU(channels),
        )
        self.residuals = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(residual_blocks))
        )
        self.post_residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.upsampling = nn.Sequential(
            UpsampleBlock(channels),
            UpsampleBlock(channels),
        )
        self.output = nn.Sequential(
            nn.Conv2d(channels, 3, kernel_size=9, padding=4),
            nn.Tanh(),
        )

    def forward(self, low_resolution: Tensor) -> Tensor:
        """Apply residual learning and two 2× stages for total 4× enlargement."""
        initial_features = self.initial(low_resolution)
        residual_features = self.post_residual(self.residuals(initial_features))
        features = initial_features + residual_features
        return self.output(self.upsampling(features))


def _discriminator_block(
    input_channels: int,
    output_channels: int,
    *,
    stride: int,
    batch_normalization: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not batch_normalization,
        )
    ]
    if batch_normalization:
        layers.append(nn.BatchNorm2d(output_channels))
    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    """Distinguish real and generated 128×128 images using raw output logits."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be greater than zero")
        self.features = nn.Sequential(
            _discriminator_block(
                3,
                base_channels,
                stride=1,
                batch_normalization=False,
            ),
            _discriminator_block(base_channels, base_channels, stride=2),
            _discriminator_block(base_channels, base_channels * 2, stride=1),
            _discriminator_block(base_channels * 2, base_channels * 2, stride=2),
            _discriminator_block(base_channels * 2, base_channels * 4, stride=1),
            _discriminator_block(base_channels * 4, base_channels * 4, stride=2),
            _discriminator_block(base_channels * 4, base_channels * 8, stride=1),
            _discriminator_block(base_channels * 8, base_channels * 8, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 1),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return one unbounded real/fake logit per image without sigmoid."""
        return self.classifier(self.features(images)).flatten()


class VGGFeatureExtractor(nn.Module):
    """Frozen VGG19 features for optional perceptual loss.

    SRGAN tensors are converted from ``[-1, 1]`` to ``[0, 1]`` and then
    normalized with ImageNet statistics before feature extraction.
    """

    def __init__(self, *, pretrained: bool = True, feature_layer: int = 36) -> None:
        super().__init__()
        weights = VGG19_Weights.DEFAULT if pretrained else None
        features = vgg19(weights=weights).features
        if not 0 < feature_layer <= len(features):
            raise ValueError("feature_layer is outside the VGG19 feature range")
        self.features = features[:feature_layer].eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.register_buffer(
            "mean",
            Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def train(self, mode: bool = True) -> VGGFeatureExtractor:
        """Keep frozen feature layers in evaluation mode."""
        super().train(False)
        return self

    def forward(self, images: Tensor) -> Tensor:
        """Return frozen ImageNet-normalized VGG feature activations."""
        normalized = ((images + 1.0) / 2.0 - self.mean) / self.std
        return self.features(normalized)


class GeneratorLoss(nn.Module):
    """Weighted SRGAN generator objective.

    ``total = pixel_weight * L1 + adversarial_weight * BCEWithLogits +
    perceptual_weight * L1(VGG(generated), VGG(target))``.
    Perceptual loss is disabled when its weight is zero. Tests should either
    disable it or inject a small local extractor, preventing weight downloads.
    """

    def __init__(
        self,
        *,
        pixel_weight: float = 1.0,
        adversarial_weight: float = 1e-3,
        perceptual_weight: float = 0.0,
        perceptual_extractor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if min(pixel_weight, adversarial_weight, perceptual_weight) < 0:
            raise ValueError("Generator loss weights must be non-negative")
        self.pixel_weight = pixel_weight
        self.adversarial_weight = adversarial_weight
        self.perceptual_weight = perceptual_weight
        self.pixel_criterion = nn.L1Loss()
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        if perceptual_weight > 0:
            self.perceptual_extractor = (
                perceptual_extractor
                if perceptual_extractor is not None
                else VGGFeatureExtractor(pretrained=True)
            )
            self.perceptual_extractor.eval()
            for parameter in self.perceptual_extractor.parameters():
                parameter.requires_grad_(False)
        else:
            self.perceptual_extractor = None

    def forward(
        self,
        generated: Tensor,
        target: Tensor,
        discriminator_logits: Tensor,
    ) -> dict[str, Tensor]:
        """Return total and named component losses without detaching gradients."""
        pixel = self.pixel_criterion(generated, target)
        adversarial = self.adversarial_criterion(
            discriminator_logits,
            discriminator_logits.new_ones(discriminator_logits.shape),
        )
        perceptual = generated.new_zeros(())
        if self.perceptual_extractor is not None:
            generated_features = self.perceptual_extractor(generated)
            with torch.no_grad():
                target_features = self.perceptual_extractor(target)
            perceptual = self.pixel_criterion(generated_features, target_features)
        total = (
            self.pixel_weight * pixel
            + self.adversarial_weight * adversarial
            + self.perceptual_weight * perceptual
        )
        return {
            "total": total,
            "pixel": pixel,
            "adversarial": adversarial,
            "perceptual": perceptual,
        }
