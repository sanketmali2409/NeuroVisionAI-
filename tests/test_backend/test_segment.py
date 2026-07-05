"""
Integration tests for the /segment endpoint.
"""

import io
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.core.app_factory import create_app

app = create_app()
client = TestClient(app)


def test_segment_endpoint_success():
    """Test successful generation of a segmentation mask."""
    from PIL import Image
    
    img = Image.new('RGB', (10, 10), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    with patch("backend.api.routers.segment.generate_segmentation") as mock_seg:
        mock_seg.return_value = {
            "mask_base64": "dummy_base64_mask=="
        }
        
        response = client.post(
            "/api/v1/segment",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["mask_base64"] == "dummy_base64_mask=="


def test_segment_endpoint_invalid_image():
    """Test segmentation endpoint with an invalid image file."""
    invalid_bytes = b"not an image file"
    
    with patch("segmentation.inference.segmenter.is_available", return_value=True):
        response = client.post(
            "/api/v1/segment",
            files={"file": ("test.txt", invalid_bytes, "text/plain")}
        )
        
        assert response.status_code == 400
        assert response.json()["error"] == "InvalidImageError"


def test_segment_endpoint_model_unavailable():
    """Test behavior when segmentation model is unavailable."""
    from PIL import Image
    
    img = Image.new('RGB', (10, 10), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    with patch("segmentation.inference.segmenter.is_available", return_value=False):
        response = client.post(
            "/api/v1/segment",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 503
        assert response.json()["error"] == "ModelNotLoadedError"
