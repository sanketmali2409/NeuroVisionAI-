"""
Pydantic schemas for the classification (predict) endpoints.
"""

from pydantic import BaseModel, Field


class PredictionDetail(BaseModel):
    class_name: str = Field(..., description="The predicted class name (e.g., glioma, meningioma).")
    confidence: float = Field(..., description="Confidence score between 0 and 1.")


class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""
    
    prediction: PredictionDetail = Field(..., description="The model's classification prediction.")
