"""Aspect-preserving center crop + resize (FluxRT behavior, PIL instead of OpenCV)."""

from __future__ import annotations

import numpy as np
from PIL import Image


def crop_maximal_rectangle(
    image: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """
    Crop maximal rectangle with target aspect ratio centered on the long axis,
    then resize to (target_width, target_height) without non-uniform stretch.
    Returns ``uint8`` RGB with shape ``(target_height, target_width, 3)`` (alpha dropped).
    """
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim != 3:
        raise ValueError(f"image must be HxW, HxWxC, or HxWx3/4; got ndim={image.ndim}")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.shape[2] != 3:
        raise ValueError(f"image must have 1, 3, or 4 channels after reshape; got C={image.shape[2]}")

    aspect_ratio = target_width / float(target_height)

    input_image_height, input_image_width = image.shape[:2]
    input_aspect_ratio = input_image_width / float(input_image_height)

    if aspect_ratio > input_aspect_ratio:
        crop_height = int(round(input_image_width / aspect_ratio))
        crop_height = max(1, min(input_image_height, crop_height))
        crop_width = input_image_width
    else:
        crop_width = int(round(input_image_height * aspect_ratio))
        crop_width = max(1, min(input_image_width, crop_width))
        crop_height = input_image_height

    start_x = (input_image_width - crop_width) // 2
    start_y = (input_image_height - crop_height) // 2

    cropped = image[start_y : start_y + crop_height, start_x : start_x + crop_width]
    pil = Image.fromarray(cropped)
    pil = pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return np.asarray(pil)
