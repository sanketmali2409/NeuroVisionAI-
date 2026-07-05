"""
Integration tests for the /predict endpoint.
"""

import io
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.core.app_factory import create_app

app = create_app()
client = TestClient(app)


def test_predict_endpoint_success():
    """Test successful generation of a prediction."""
    from PIL import Image
    
    img = Image.new('RGB', (10, 10), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    with patch("backend.api.routers.predict.process_prediction") as mock_pred:
        mock_pred.return_value = {
            "prediction": {"class_name": "glioma", "confidence": 0.99}
        }
        
        response = client.post(
            "/api/v1/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"]["class_name"] == "glioma"
        assert data["prediction"]["confidence"] == 0.99


def test_predict_endpoint_invalid_image():
    """Test prediction endpoint with an invalid image file."""
    invalid_bytes = b"not an image file"
    
    with patch("classification.inference.classifier.is_available", return_value=True):
        response = client.post(
            "/api/v1/predict",
            files={"file": ("test.txt", invalid_bytes, "text/plain")}
        )
        
        assert response.status_code == 400
        assert response.json()["error"] == "InvalidImageError"


def test_predict_endpoint_model_unavailable():
    """Test behavior when classification model is unavailable."""
    from PIL import Image
    
    img = Image.new('RGB', (10, 10), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    with patch("classification.inference.classifier.is_available", return_value=False):
        response = client.post(
            "/api/v1/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 503
        assert response.json()["error"] == "ModelNotLoadedError"
