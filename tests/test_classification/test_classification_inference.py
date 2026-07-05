"""
Unit tests for the Classification module.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np

# Mock out tensorflow before running tests
sys.modules['tensorflow'] = MagicMock()

from classification.inference import classifier


def test_classifier_is_available(tmp_path):
    """Test availability check relies on file existence."""
    # Temporarily change model path to a file we know doesn't exist
    classifier.model_path = tmp_path / "nonexistent.h5"
    assert classifier.is_available() is False
    
    # Create the file and check again
    classifier.model_path.touch()
    assert classifier.is_available() is True


def test_classifier_lazy_loading():
    """Test that model is loaded only when requested."""
    import tensorflow as tf
    
    # Reset internal model
    classifier._model = None
    
    with patch.object(classifier, 'is_available', return_value=True):
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_load.return_value = "mocked_model_instance"
            
            # Access property
            m = classifier.model
            
            assert m == "mocked_model_instance"
            mock_load.assert_called_once()
            
            # Second access shouldn't reload
            m2 = classifier.model
            assert mock_load.call_count == 1


def test_classifier_predict():
    """Test inference parsing logic without TF installed."""
    
    img_array = np.random.rand(1, 224, 224, 3)
    
    mock_model = MagicMock()
    # Mock predict to return confidence for 4 classes (glioma, meningioma, no_tumor, pituitary)
    # Let's say it predicts meningioma (index 1) with high confidence
    mock_model.predict.return_value = np.array([[0.1, 0.8, 0.05, 0.05]])
    
    classifier._model = mock_model
    
    class_name, confidence = classifier.predict(img_array)
    
    assert class_name == "meningioma"
    assert confidence == 0.8
    mock_model.predict.assert_called_once_with(img_array, verbose=0)
