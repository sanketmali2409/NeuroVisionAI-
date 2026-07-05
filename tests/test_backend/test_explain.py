"""
Integration tests for the /explain endpoint.
"""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.core.app_factory import create_app
from backend.core.exceptions import InvalidImageError, ModelNotLoadedError

# We use the app factory to get a fresh instance for testing
app = create_app()
client = TestClient(app)


def test_explain_endpoint_success():
    """Test successful generation of an explanation."""
    
    # Create a small dummy image in memory
    from PIL import Image
    import io
    img = Image.new('RGB', (10, 10), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    # We mock the entire generate_explanation to avoid needing TF in HTTP test
    with patch("backend.api.routers.explain.generate_explanation") as mock_gen:
        mock_gen.return_value = {
            "method": "gradcam",
            "prediction": {"class_name": "glioma", "confidence": 0.99},
            "heatmap_base64": "dummybase64string=="
        }
        
        response = client.post(
            "/api/v1/explain",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"method": "gradcam"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["method"] == "gradcam"
        assert data["prediction"]["class_name"] == "glioma"
        assert data["heatmap_base64"] == "dummybase64string=="


def test_explain_endpoint_invalid_image():
    """Test explanation endpoint with an invalid image file."""
    
    # We let the real service handle this to test its validation, 
    # but we patch out the module loading part
    
    invalid_bytes = b"not an image file"
    
    with patch("xai.gradcam_explainer.is_available", return_value=True):
        response = client.post(
            "/api/v1/explain",
            files={"file": ("test.txt", invalid_bytes, "text/plain")}
        )
        
        assert response.status_code == 400
        assert response.json()["error"] == "InvalidImageError"


def test_explain_endpoint_model_unavailable():
    """Test behavior when explainability is unavailable."""
    
    # Create a small dummy image in memory
    from PIL import Image
    import io
    img = Image.new('RGB', (10, 10), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    with patch("xai.gradcam_explainer.is_available", return_value=False):
        response = client.post(
            "/api/v1/explain",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 503
        assert response.json()["error"] == "ModelNotLoadedError"
