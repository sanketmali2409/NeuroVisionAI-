"""
Router for the /segment endpoint.
"""

from fastapi import APIRouter, File, UploadFile, status

from backend.api.schemas.segment import SegmentationResponse
from backend.api.services.segment_service import generate_segmentation

router = APIRouter(prefix="/segment", tags=["Segmentation"])


@router.post("", response_model=SegmentationResponse, status_code=status.HTTP_200_OK)
async def segment_image(file: UploadFile = File(...)):
    """
    Generates a segmentation mask for a brain MRI scan highlighting the tumor region.
    Returns a Base64-encoded PNG image of the mask.
    """
    file_bytes = await file.read()
    
    # Delegating business logic to the service layer
    return generate_segmentation(file_bytes)
