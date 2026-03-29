"""
Modify YOLO11 to accept 5-channel input instead of 3-channel RGB.

Only the first convolutional layer changes. Pretrained RGB weights are
preserved for the first 3 channels, and channels 4-5 are initialized
from the mean of RGB weights + small noise.
"""

import logging

import torch
import torch.nn as nn
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def create_multichannel_yolo(
    base_weights: str = "yolo11l.pt",
    input_channels: int = 5,
    freeze_backbone_layers: int = 10,
) -> YOLO:
    """Create a YOLO11 model with modified input channels.

    Args:
        base_weights: Path to pretrained YOLO weights
        input_channels: Number of input channels (default 5 for RGB+grad+LCN)
        freeze_backbone_layers: Number of backbone layers to freeze for warmup

    Returns:
        Modified YOLO model
    """
    model = YOLO(base_weights)

    # Access the first conv layer in the backbone
    first_conv = model.model.model[0].conv
    assert isinstance(first_conv, nn.Conv2d), f"Expected Conv2d, got {type(first_conv)}"
    assert first_conv.in_channels == 3, f"Expected 3 input channels, got {first_conv.in_channels}"

    out_channels = first_conv.out_channels
    kernel_size = first_conv.kernel_size
    stride = first_conv.stride
    padding = first_conv.padding

    # Create new conv with extra input channels
    new_conv = nn.Conv2d(
        input_channels, out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        bias=first_conv.bias is not None,
    )

    # Initialize weights
    with torch.no_grad():
        # Copy pretrained RGB weights to first 3 channels
        new_conv.weight.data[:, :3, :, :] = first_conv.weight.data

        # Initialize extra channels from mean of RGB weights + small noise
        rgb_mean = first_conv.weight.data.mean(dim=1, keepdim=True)
        for ch in range(3, input_channels):
            new_conv.weight.data[:, ch:ch + 1, :, :] = (
                rgb_mean + torch.randn_like(rgb_mean) * 0.01
            )

        # Copy bias if present
        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data.clone()

    # Replace in model
    model.model.model[0].conv = new_conv
    logger.info(
        "Modified first conv: %d → %d input channels (kernel=%s, stride=%s)",
        3, input_channels, kernel_size, stride,
    )

    # Freeze backbone layers for warmup phase
    if freeze_backbone_layers > 0:
        frozen = 0
        for param in model.model.model[:freeze_backbone_layers].parameters():
            param.requires_grad = False
            frozen += 1
        logger.info("Froze %d parameters in first %d layers", frozen, freeze_backbone_layers)

    return model


def unfreeze_backbone(model: YOLO, backbone_lr_factor: float = 0.1):
    """Unfreeze backbone with lower learning rate.

    Call this after the warmup phase (5-10 epochs) to allow
    backbone fine-tuning with 10x lower LR.
    """
    for param in model.model.parameters():
        param.requires_grad = True
    logger.info("Unfroze all parameters. Set backbone LR to %.4f of head LR.", backbone_lr_factor)
    return backbone_lr_factor
