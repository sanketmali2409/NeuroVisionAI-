"""
Pydantic schemas for the segmentation endpoints.
"""

from pydantic import BaseModel, Field


class SegmentationResponse(BaseModel):
    """Response schema for the /segment endpoint."""
    
    mask_base64: str = Field(..., description="Base64 encoded PNG of the segmentation mask.")
