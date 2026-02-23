"""
EPI Check project — Pydantic schemas with OpenAPI documentation.
"""
from typing import Optional
from pydantic import BaseModel, Field


class PPEConfig(BaseModel):
    """Configure which PPE classes are active for detection and training."""
    thermal_coat: bool = Field(False, description="Enable thermal coat detection (class ID 0)")
    thermal_pants: bool = Field(False, description="Enable thermal pants detection (class ID 1)")
    gloves: bool = Field(False, description="Enable gloves detection (class ID 2)")
    helmet: bool = Field(True, description="Enable helmet detection (class ID 3)")
    boots: bool = Field(True, description="Enable boots detection (class ID 4)")
    person: bool = Field(True, description="Enable person detection (class ID 5)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "thermal_coat": False,
                    "thermal_pants": False,
                    "gloves": False,
                    "helmet": True,
                    "boots": True,
                    "person": True,
                }
            ]
        }
    }


class DetectRequest(BaseModel):
    """Request body for base64 image detection via REST API."""
    image_base64: Optional[str] = Field(
        None, description="Base64-encoded image (JPEG or PNG). "
                          "Supports data URI prefix: `data:image/jpeg;base64,...`"
    )
    image_url: Optional[str] = Field(
        None, description="URL to fetch image from (alternative to base64)"
    )
    camera_id: Optional[int] = Field(
        None, description="Camera identifier for MQTT alert routing", examples=[1]
    )
    zone_id: Optional[int] = Field(
        None, description="Zone identifier for location-based reporting", examples=[3]
    )
    model_name: str = Field(
        "best", description="Model name to use for detection.",
        examples=["best", "last"]
    )
    confidence: float = Field(
        0.4, ge=0.1, le=0.95,
        description="PPE detection confidence threshold.",
        examples=[0.45]
    )
    detect_faces: bool = Field(
        False, description="Enable InsightFace face recognition."
    )
    face_threshold: float = Field(
        0.45, ge=0.1, le=0.9,
        description="Face cosine similarity threshold.",
        examples=[0.45]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "image_base64": "/9j/4AAQSkZJRg...",
                    "model_name": "best",
                    "confidence": 0.45,
                    "detect_faces": True,
                    "face_threshold": 0.45,
                    "camera_id": 1,
                    "zone_id": 3,
                }
            ]
        }
    }


class DetectionItem(BaseModel):
    """Single detected object (PPE item or person)."""
    class_name: str = Field(..., description="Detected class name", examples=["helmet"])
    confidence: float = Field(..., description="Detection confidence 0.0–1.0", examples=[0.92])
    bbox: dict = Field(
        {}, description="Bounding box in pixels: {x, y, w, h}",
        examples=[{"x": 150, "y": 80, "w": 45, "h": 35}]
    )


class FaceResult(BaseModel):
    """Face recognition result for a detected face."""
    recognized: bool = Field(False, description="True if face matched a registered person")
    person_name: str = Field("UNKNOWN", description="Display name of matched person")
    person_code: str = Field("", description="Unique person code")
    confidence: float = Field(0.0, description="Cosine similarity score 0.0–1.0")
    bbox: dict = Field({}, description="Face bounding box in pixels: {x, y, w, h}")


class DetectResponse(BaseModel):
    """Full detection response."""
    compliant: bool = Field(False, description="True if all required PPE items were detected")
    required_count: int = Field(0, description="Number of required PPE classes")
    detected_count: int = Field(0, description="Number of required PPE classes actually detected")
    missing: list[str] = Field([], description="List of missing PPE class names")
    detections: list[DetectionItem] = Field([], description="All detected objects with bounding boxes")
    faces: list[FaceResult] = Field([], description="Face recognition results")
    model_name: str = Field("", description="Model used for detection")
    processing_ms: int = Field(0, description="Total processing time in milliseconds")
    snapshot_url: Optional[str] = Field(None, description="URL to download annotated result image")
    annotated_base64: Optional[str] = Field(None, description="Base64-encoded annotated image")


class TrainRequest(BaseModel):
    """Configuration for starting a YOLOv8 training run."""
    base_model: str = Field(
        "yolov8m.pt", pattern=r"^yolov8[nslmx]\.pt$",
        description="Pre-trained base model.",
        examples=["yolov8m.pt"]
    )
    epochs: int = Field(60, ge=10, le=300, examples=[60])
    batch_size: int = Field(16, ge=4, le=64, examples=[16])
    img_size: int = Field(640, ge=320, le=1280, examples=[640])
    patience: int = Field(15, ge=5, examples=[15])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "base_model": "yolov8m.pt",
                    "epochs": 60,
                    "batch_size": 16,
                    "img_size": 640,
                    "patience": 15,
                }
            ]
        }
    }


class VideoSource(BaseModel):
    """Video source configuration."""
    source_type: str = Field(
        ..., pattern="^(file|rtsp|youtube|webcam)$",
        examples=["rtsp"]
    )
    url: str = Field("", examples=["rtsp://admin:pass@192.168.1.100:554/stream"])
    camera_id: Optional[int] = Field(None)
    zone_id: Optional[int] = Field(None)
    model_name: str = Field("best")
    confidence: float = Field(0.4)
    detect_faces: bool = Field(False)


class ConvertRequest(BaseModel):
    """Configuration for annotation format conversion."""
    remap: dict = Field({}, examples=[{"0": 3, "1": 4}])
    source_dir: str = Field("raw_labels")


class AnnotationSave(BaseModel):
    """Save YOLO bounding box annotations for an image."""
    image_filename: str = Field(..., examples=["photo_001.jpg"])
    annotations: list[dict] = Field(
        [],
        examples=[[
            {"class_id": 3, "cx": 0.447, "cy": 0.333, "w": 0.067, "h": 0.046},
            {"class_id": 5, "cx": 0.500, "cy": 0.500, "w": 0.400, "h": 0.900},
        ]]
    )


class FaceRegisterRequest(BaseModel):
    """Register a person for face recognition."""
    person_code: str = Field(..., max_length=50, examples=["carlos_santos"])
    person_name: str = Field(..., max_length=255, examples=["Carlos Santos"])
    badge_id: str = Field("", examples=["EMP001"])
    image_base64: Optional[str] = Field(None)


class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = Field(True)
    message: str = Field("OK")
    data: Optional[dict | list] = Field(None)
    company_id: Optional[int] = Field(None, examples=[1])
