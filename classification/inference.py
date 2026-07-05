"""
Classification inference wrapper using EfficientNetB0.

TensorFlow and heavy ML libraries are imported lazily to ensure fast
application startup and to follow the project's architectural conventions.
"""

from __future__ import annotations

import os
from pathlib import Path

from config.logging_config import get_logger

logger = get_logger(__name__)

# Default mapping for model output classes
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]


class ClassifierWrapper:
    """Wrapper for loading and running predictions with the trained classifier."""

    def __init__(self, model_path: str = "models/saved_models/classification/efficientnetb0.h5") -> None:
        self.model_path = Path(model_path)
        self._model = None

    def is_available(self) -> bool:
        """Check if the trained model artifact exists on disk."""
        return self.model_path.exists()

    @property
    def model(self):
        """Lazy load the TensorFlow model."""
        if self._model is None:
            if not self.is_available():
                logger.warning(f"Classification model not found at {self.model_path}")
                return None
                
            logger.info(f"Loading classification model from {self.model_path}")
            import tensorflow as tf
            # Load the pre-trained model
            self._model = tf.keras.models.load_model(str(self.model_path), compile=False)
        return self._model

    def predict(self, img_array) -> tuple[str, float]:
        """
        Runs inference on the provided preprocessed image array.
        
        Args:
            img_array: Numpy array of shape (1, 224, 224, 3) normalized [0, 1]
            
        Returns:
            Tuple of (predicted_class_name, confidence_score)
        """
        import numpy as np
        
        model = self.model
        if model is None:
            raise RuntimeError("Model is not loaded or available.")

        preds = model.predict(img_array, verbose=0)
        pred_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][pred_idx])
        
        # Guard against index out of bounds if model has different num classes
        if pred_idx < len(CLASS_NAMES):
            class_name = CLASS_NAMES[pred_idx]
        else:
            class_name = f"class_{pred_idx}"

        return class_name, confidence


# Module-level singleton
classifier = ClassifierWrapper()
