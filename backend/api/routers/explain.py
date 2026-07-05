"""
Router for the /explain endpoint.
"""

from fastapi import APIRouter, File, Form, UploadFile, status

from backend.api.schemas.explain import ExplainResponse
from backend.api.services.explain_service import generate_explanation

router = APIRouter(prefix="/explain", tags=["Explainability"])


@router.post("", response_model=ExplainResponse, status_code=status.HTTP_200_OK)
async def explain_image(
    file: UploadFile = File(...),
    method: str = Form(default="gradcam", description="Explainability method: 'gradcam' or 'gradcam++'")
):
    """
    Generates a heatmap explanation (Grad-CAM or Grad-CAM++) for a brain MRI scan.
    
    The API returns the original model prediction and a base64-encoded JPEG image
    of the heatmap superimposed on the original scan.
    """
    file_bytes = await file.read()
    
    # Delegating business logic to the service layer
    return generate_explanation(file_bytes, method=method)
