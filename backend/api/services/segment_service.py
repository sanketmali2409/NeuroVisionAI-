"""
Business logic for the /segment endpoint.
"""

import base64
import io

from PIL import Image, UnidentifiedImageError

from backend.api.schemas.segment import SegmentationResponse
from backend.core.exceptions import InvalidImageError, ModelNotLoadedError
from config.logging_config import get_logger
from segmentation.inference import segmenter

logger = get_logger(__name__)


def generate_segmentation(file_bytes: bytes) -> SegmentationResponse:
    """Generates a segmentation mask for the given image."""
    
    if not segmenter.is_available():
        raise ModelNotLoadedError("Segmentation model is not available.")

    try:
        # Verify it's a valid image
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        logger.warning(f"Invalid image format: {e}")
        raise InvalidImageError("The uploaded file could not be parsed as a valid image.")

    # Preprocess image for U-Net (assuming 256x256)
    import numpy as np
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    try:
        mask = segmenter.predict(img_tensor)
    except Exception as e:
        logger.error(f"Error generating segmentation mask: {e}")
        raise ModelNotLoadedError(f"Failed to generate mask: {e}")

    # Convert binary mask (0s and 1s) to an image (0 and 255)
    mask_img_array = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_img_array, mode="L")
    
    # Base64 encode the PNG
    img_byte_arr = io.BytesIO()
    mask_img.save(img_byte_arr, format="PNG")
    mask_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    return SegmentationResponse(mask_base64=mask_base64)
