"""
Business logic for the /explain endpoint.
"""

import base64
import io

from PIL import Image, UnidentifiedImageError

from backend.api.schemas.explain import ExplainResponse, PredictionDetail
from backend.core.exceptions import InvalidImageError, ModelNotLoadedError
from config.logging_config import get_logger
from xai import gradcam_explainer

logger = get_logger(__name__)


def generate_explanation(file_bytes: bytes, method: str = "gradcam") -> ExplainResponse:
    """Generates an explainability heatmap for the given image."""
    
    if not gradcam_explainer.is_available():
        raise ModelNotLoadedError("Explainability module is not available.")

    if method not in ("gradcam", "gradcam++"):
        # Default to gradcam if unknown
        method = "gradcam"

    try:
        # Verify it's a valid image
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        logger.warning(f"Invalid image format: {e}")
        raise InvalidImageError("The uploaded file could not be parsed as a valid image.")

    from classification.inference import classifier
    if not classifier.is_available():
        raise ModelNotLoadedError("Classification model is not loaded yet.")
        
    model = classifier.model
    last_conv_layer = "top_activation" # Usually top_activation for EfficientNetB0

    # Preprocess image for the model
    # (assuming 224x224 as it's standard for EfficientNetB0)
    import numpy as np
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    # Get actual prediction
    class_name, confidence = classifier.predict(img_tensor)

    # Generate the heatmap using the specified method
    try:
        heatmap = gradcam_explainer.generate_heatmap(
            img_array=img_tensor, 
            model=model, 
            last_conv_layer_name=last_conv_layer,
            method=method
        )
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        raise ModelNotLoadedError(f"Failed to generate heatmap: {e}")

    # Overlay heatmap on original image
    overlay_bytes = gradcam_explainer.overlay_heatmap(img_array, heatmap)
    
    # Base64 encode
    heatmap_base64 = base64.b64encode(overlay_bytes).decode("utf-8")

    prediction = PredictionDetail(class_name=class_name, confidence=confidence)

    return ExplainResponse(
        method=method,
        prediction=prediction,
        heatmap_base64=heatmap_base64
    )
