"""
Business logic for the /predict endpoint.
"""

import io

from PIL import Image, UnidentifiedImageError

from backend.api.schemas.predict import PredictionDetail, PredictionResponse
from backend.core.exceptions import InvalidImageError, ModelNotLoadedError
from classification.inference import classifier
from config.logging_config import get_logger

logger = get_logger(__name__)


def process_prediction(file_bytes: bytes) -> PredictionResponse:
    """Processes an image for classification and returns the prediction."""
    
    if not classifier.is_available():
        raise ModelNotLoadedError("Classification model is not available.")

    try:
        # Verify it's a valid image
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        logger.warning(f"Invalid image format: {e}")
        raise InvalidImageError("The uploaded file could not be parsed as a valid image.")

    # Preprocess image for EfficientNetB0
    import numpy as np
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    try:
        class_name, confidence = classifier.predict(img_tensor)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise ModelNotLoadedError(f"Failed to run prediction: {e}")

    prediction = PredictionDetail(class_name=class_name, confidence=confidence)

    return PredictionResponse(prediction=prediction)
