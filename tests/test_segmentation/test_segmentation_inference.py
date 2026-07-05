"""
Unit tests for the Segmentation module.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np

# Mock out tensorflow before running tests
sys.modules['tensorflow'] = MagicMock()

from segmentation.inference import segmenter


def test_segmenter_is_available(tmp_path):
    """Test availability check relies on file existence."""
    # Temporarily change model path to a file we know doesn't exist
    segmenter.model_path = tmp_path / "nonexistent.h5"
    assert segmenter.is_available() is False
    
    # Create the file and check again
    segmenter.model_path.touch()
    assert segmenter.is_available() is True


def test_segmenter_lazy_loading():
    """Test that model is loaded only when requested."""
    import tensorflow as tf
    
    # Reset internal model
    segmenter._model = None
    
    with patch.object(segmenter, 'is_available', return_value=True):
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_load.return_value = "mocked_unet_instance"
            
            # Access property
            m = segmenter.model
            
            assert m == "mocked_unet_instance"
            mock_load.assert_called_once()
            
            # Second access shouldn't reload
            m2 = segmenter.model
            assert mock_load.call_count == 1


def test_segmenter_predict():
    """Test inference parsing logic without TF installed."""
    
    img_array = np.random.rand(1, 256, 256, 3)
    
    mock_model = MagicMock()
    
    # Mock predict to return a dummy mask probabilities (shape: 1, 256, 256, 1)
    # Let's make half of it > 0.5 and half < 0.5
    dummy_probs = np.zeros((1, 256, 256, 1))
    dummy_probs[:, :128, :, :] = 0.9  # Tumor region
    dummy_probs[:, 128:, :, :] = 0.1  # Normal region
    
    mock_model.predict.return_value = dummy_probs
    
    segmenter._model = mock_model
    
    mask = segmenter.predict(img_array)
    
    assert mask.shape == (256, 256)
    assert np.all(mask[:128, :] == 1)
    assert np.all(mask[128:, :] == 0)
    mock_model.predict.assert_called_once_with(img_array, verbose=0)
