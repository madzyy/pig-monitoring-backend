from pydantic import BaseModel
from typing import List, Any

class DetectionResponse(BaseModel):
    id: str
    filename: str
    confidence: float
    bbox: List[float]
    class_id: int  # 0=lying, 1=sleeping, 2=investigating, 3=eating, 4=walking, 5=mounted
    timestamp: Any

    class Config:
        from_attributes = True  # Updated for Pydantic v2

class PredictionResponse(BaseModel):
    filename: str
    detections: List[DetectionResponse]
