"""
Router for the /predict endpoint.
"""

from fastapi import APIRouter, File, UploadFile, status

from backend.api.schemas.predict import PredictionResponse
from backend.api.services.predict_service import process_prediction

router = APIRouter(prefix="/predict", tags=["Classification"])


@router.post("", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_image(file: UploadFile = File(...)):
    """
    Classifies a brain MRI scan into one of four categories:
    glioma, meningioma, pituitary, or no_tumor.
    """
    file_bytes = await file.read()
    
    # Delegating business logic to the service layer
    return process_prediction(file_bytes)
