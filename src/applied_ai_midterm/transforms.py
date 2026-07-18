"""Image transformations for classification and paired super-resolution data."""

from __future__ import annotations

from collections.abc import Callable

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    Resize,
    ToTensor,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SRGAN_MEAN = (0.5, 0.5, 0.5)
SRGAN_STD = (0.5, 0.5, 0.5)


def classifier_transform(
    image_size: int = 128,
    *,
    training: bool,
) -> Callable[[Image.Image], Tensor]:
    """Build an ImageNet-normalized classifier transform.

    Training uses random crop, flip, moderate rotation, and color jitter.
    Evaluation performs only deterministic resizing and normalization.
    """
    if image_size <= 0:
        raise ValueError("image_size must be greater than zero")

    if training:
        spatial_and_color = [
            RandomResizedCrop(
                image_size,
                scale=(0.80, 1.0),
                ratio=(0.90, 1.10),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
            ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        ]
    else:
        spatial_and_color = [
            Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        ]

    return Compose(
        [*spatial_and_color, ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


class SRGANPairTransform:
    """Create aligned low/high-resolution tensors normalized to ``[-1, 1]``.

    Spatial augmentation is applied once to the high-resolution target. The
    low-resolution input is then bicubically downsampled from that target, so
    both tensors remain exactly aligned. Evaluation mode is deterministic.
    """

    def __init__(
        self,
        low_resolution_size: int = 32,
        high_resolution_size: int = 128,
        *,
        training: bool,
    ) -> None:
        if low_resolution_size <= 0 or high_resolution_size <= 0:
            raise ValueError("SRGAN image sizes must be greater than zero")
        if high_resolution_size != low_resolution_size * 4:
            raise ValueError("high_resolution_size must be 4x low_resolution_size")
        self.low_resolution_size = low_resolution_size
        self.high_resolution_size = high_resolution_size
        self.training = training
        self.color_jitter = ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.08,
            hue=0.01,
        )

    def __call__(self, image: Image.Image) -> tuple[Tensor, Tensor]:
        """Return a bicubic low-resolution input and aligned real target."""
        high_resolution = image.convert("RGB")
        if self.training:
            top, left, height, width = RandomResizedCrop.get_params(
                high_resolution,
                scale=(0.80, 1.0),
                ratio=(0.90, 1.10),
            )
            high_resolution = F.resized_crop(
                high_resolution,
                top,
                left,
                height,
                width,
                [self.high_resolution_size, self.high_resolution_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            if torch.rand(1).item() < 0.5:
                high_resolution = F.hflip(high_resolution)
            angle = RandomRotation.get_params([-10.0, 10.0])
            high_resolution = F.rotate(
                high_resolution,
                angle,
                interpolation=InterpolationMode.BILINEAR,
            )
            high_resolution = self.color_jitter(high_resolution)
        else:
            high_resolution = F.resize(
                high_resolution,
                [self.high_resolution_size, self.high_resolution_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )

        low_resolution = F.resize(
            high_resolution,
            [self.low_resolution_size, self.low_resolution_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        to_tensor = ToTensor()
        normalize = Normalize(SRGAN_MEAN, SRGAN_STD)
        return normalize(to_tensor(low_resolution)), normalize(
            to_tensor(high_resolution)
        )


def denormalize_classifier(tensor: Tensor) -> Tensor:
    """Reverse ImageNet normalization and clamp pixels for display."""
    return _reverse_normalization(tensor, IMAGENET_MEAN, IMAGENET_STD)


def denormalize_srgan(tensor: Tensor) -> Tensor:
    """Map an SRGAN tensor from ``[-1, 1]`` back to display range ``[0, 1]``."""
    return _reverse_normalization(tensor, SRGAN_MEAN, SRGAN_STD)


def _reverse_normalization(
    tensor: Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> Tensor:
    if tensor.ndim not in {3, 4} or tensor.shape[-3] != 3:
        raise ValueError("Expected an RGB tensor with shape (3,H,W) or (N,3,H,W)")
    shape = [1] * tensor.ndim
    shape[-3] = 3
    means = tensor.new_tensor(mean).view(shape)
    standard_deviations = tensor.new_tensor(std).view(shape)
    return (tensor * standard_deviations + means).clamp(0.0, 1.0)
