"""
Segmentation inference wrapper using U-Net.

TensorFlow and heavy ML libraries are imported lazily to ensure fast
application startup and to follow the project's architectural conventions.
"""

from __future__ import annotations

from pathlib import Path

from config.logging_config import get_logger

logger = get_logger(__name__)


class SegmenterWrapper:
    """Wrapper for loading and running predictions with the trained U-Net segmentation model."""

    def __init__(self, model_path: str = "models/saved_models/segmentation/unet.h5") -> None:
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
                logger.warning(f"Segmentation model not found at {self.model_path}")
                return None
                
            logger.info(f"Loading segmentation model from {self.model_path}")
            import tensorflow as tf
            # Load the pre-trained model
            self._model = tf.keras.models.load_model(str(self.model_path), compile=False)
        return self._model

    def predict(self, img_array):
        """
        Runs inference on the provided preprocessed image array.
        
        Args:
            img_array: Numpy array of shape (1, 256, 256, 3) normalized [0, 1]
            
        Returns:
            A binary numpy mask array of shape (256, 256).
        """
        import numpy as np
        
        model = self.model
        if model is None:
            raise RuntimeError("Model is not loaded or available.")

        # Predict the mask
        preds = model.predict(img_array, verbose=0)
        
        # Squeeze out batch and channel dimensions if necessary
        mask = np.squeeze(preds[0])
        
        # Binarize mask
        binary_mask = (mask > 0.5).astype(np.uint8)

        return binary_mask


# Module-level singleton
segmenter = SegmenterWrapper()
