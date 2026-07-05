"""
Pydantic schemas for the explainability endpoints.
"""

from pydantic import BaseModel, Field


class PredictionDetail(BaseModel):
    class_name: str = Field(..., description="The predicted class name.")
    confidence: float = Field(..., description="Confidence score between 0 and 1.")


class ExplainResponse(BaseModel):
    """Response schema for the /explain endpoint."""
    
    method: str = Field(..., description="The explainability method used (e.g., 'gradcam').")
    prediction: PredictionDetail = Field(..., description="The original model prediction.")
    heatmap_base64: str = Field(..., description="Base64 encoded JPEG of the superimposed heatmap.")
