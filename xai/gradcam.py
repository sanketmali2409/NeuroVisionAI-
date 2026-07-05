"""
Grad-CAM and Grad-CAM++ implementations.

TensorFlow and heavy ML libraries are imported lazily to ensure fast
application startup and to follow the project's architectural conventions.
"""

from __future__ import annotations

import base64
import io
import logging

from config.logging_config import get_logger

logger = get_logger(__name__)


class GradCAMWrapper:
    """Wrapper for generating Grad-CAM and Grad-CAM++ heatmaps."""

    def __init__(self) -> None:
        self._is_available = True  # Can be toggled if dependencies fail to load

    def is_available(self) -> bool:
        return self._is_available

    def generate_heatmap(
        self, img_array, model, last_conv_layer_name: str, pred_index: int | None = None, method: str = "gradcam"
    ):
        """Generates a Grad-CAM or Grad-CAM++ heatmap.
        
        Args:
            img_array: Preprocessed image array of shape (1, H, W, C).
            model: tf.keras.Model instance.
            last_conv_layer_name: Name of the last convolutional layer in the model.
            pred_index: Index of the target class. If None, the predicted class is used.
            method: "gradcam" or "gradcam++".
        """
        import numpy as np
        import tensorflow as tf
        
        # Create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            model.inputs, 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # Gradients of the top predicted class wrt the output feature map
        grads = tape.gradient(class_channel, last_conv_layer_output)

        if method == "gradcam++":
            # Grad-CAM++ weights
            first_derivative = tf.exp(class_channel) * grads
            second_derivative = tf.exp(class_channel) * grads * grads
            third_derivative = tf.exp(class_channel) * grads * grads * grads

            global_sum = tf.reduce_sum(last_conv_layer_output, axis=(0, 1, 2))
            alpha_num = second_derivative
            alpha_denom = second_derivative * 2.0 + third_derivative * global_sum
            alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
            alphas = alpha_num / alpha_denom

            weights = tf.maximum(first_derivative, 0.0)
            alphas_thresholding = tf.where(weights, alphas, 0.0)
            
            alpha_normalization_constant = tf.reduce_sum(alphas_thresholding, axis=(0, 1))
            alpha_normalization_constant = tf.where(
                alpha_normalization_constant != 0.0, alpha_normalization_constant, tf.ones_like(alpha_normalization_constant)
            )
            alphas /= tf.reshape(alpha_normalization_constant, [1, 1, -1])
            
            deep_linearization_weights = tf.reduce_sum((weights * alphas), axis=(1, 2))
            pooled_grads = deep_linearization_weights
        else:
            # Standard Grad-CAM weights: mean pooling of gradients over spatial dimensions
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def overlay_heatmap(self, img_array, heatmap, alpha: float = 0.4) -> bytes:
        """Superimposes the heatmap on the original image and returns as bytes."""
        import numpy as np
        import tensorflow as tf
        import matplotlib.cm as cm
        from PIL import Image

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
        jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img_array
        superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        superimposed_img.save(img_byte_arr, format="JPEG")
        return img_byte_arr.getvalue()


# Module-level singleton
gradcam_explainer = GradCAMWrapper()
