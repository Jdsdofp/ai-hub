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
        "best", description="Model name to use for detection. Use 'best' for the latest trained model.",
        examples=["best", "last"]
    )
    confidence: float = Field(
        0.4, ge=0.1, le=0.95,
        description="PPE detection confidence threshold. Higher = fewer false positives, lower = fewer missed detections. "
                    "Recommended: 0.45 for balanced, 0.60+ if getting false positives.",
        examples=[0.45]
    )
    detect_faces: bool = Field(
        False, description="Enable InsightFace face recognition. "
                           "Requires registered people in /faces/register."
    )
    face_threshold: float = Field(
        0.45, ge=0.1, le=0.9,
        description="Face cosine similarity threshold. Lower = more lenient matching, "
                    "higher = stricter matching. Range: 0.30 (lenient) to 0.60 (strict).",
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
        {}, description="Bounding box in pixels: {x, y, w, h} where (x,y) is top-left corner",
        examples=[{"x": 150, "y": 80, "w": 45, "h": 35}]
    )


class FaceResult(BaseModel):
    """Face recognition result for a detected face."""
    recognized: bool = Field(False, description="True if face matched a registered person")
    person_name: str = Field("UNKNOWN", description="Display name of matched person", examples=["Carlos Santos"])
    person_code: str = Field("", description="Unique person code", examples=["carlos_santos"])
    confidence: float = Field(0.0, description="Cosine similarity score 0.0–1.0", examples=[0.87])
    bbox: dict = Field(
        {}, description="Face bounding box in pixels: {x, y, w, h}",
        examples=[{"x": 160, "y": 60, "w": 50, "h": 60}]
    )


class DetectResponse(BaseModel):
    """Full detection response with compliance status, PPE detections, and face results."""
    compliant: bool = Field(False, description="True if all required PPE items were detected")
    required_count: int = Field(0, description="Number of required PPE classes (excludes 'person')")
    detected_count: int = Field(0, description="Number of required PPE classes actually detected")
    missing: list[str] = Field([], description="List of missing PPE class names", examples=[["boots"]])
    detections: list[DetectionItem] = Field([], description="All detected objects with bounding boxes")
    faces: list[FaceResult] = Field([], description="Face recognition results (if detect_faces=true)")
    model_name: str = Field("", description="Model used for detection")
    processing_ms: int = Field(0, description="Total processing time in milliseconds", examples=[145])
    snapshot_url: Optional[str] = Field(
        None, description="URL to download annotated result image",
        examples=["/api/v1/epi/results/result_a1b2c3d4.jpg?company_id=1"]
    )
    annotated_base64: Optional[str] = Field(
        None, description="Base64-encoded annotated image with bounding boxes drawn"
    )


class TrainRequest(BaseModel):
    """Configuration for starting a YOLOv8 training run."""
    base_model: str = Field(
        "yolov8m.pt", pattern=r"^yolov8[nslmx]\.pt$",
        description="Pre-trained base model. Options: yolov8n.pt (fast), yolov8s.pt (small), "
                    "yolov8m.pt (medium, recommended), yolov8l.pt (large), yolov8x.pt (extra large).",
        examples=["yolov8m.pt"]
    )
    epochs: int = Field(
        60, ge=10, le=300,
        description="Number of training epochs. More epochs = better accuracy but longer training.",
        examples=[60]
    )
    batch_size: int = Field(
        16, ge=4, le=64,
        description="Batch size. Reduce if GPU runs out of memory (T4=16, RTX 3090=32).",
        examples=[16]
    )
    img_size: int = Field(
        640, ge=320, le=1280,
        description="Input image resolution in pixels. 640 is standard, 416 for faster training.",
        examples=[640]
    )
    patience: int = Field(
        15, ge=5,
        description="Early stopping patience. Training stops if no improvement for N epochs.",
        examples=[15]
    )

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
    """Video source configuration for streaming or batch processing."""
    source_type: str = Field(
        ..., pattern="^(file|rtsp|youtube|webcam)$",
        description="Source type: file, rtsp, youtube, or webcam",
        examples=["rtsp"]
    )
    url: str = Field(
        "", description="Source URL or path. RTSP: rtsp://user:pass@ip:554/stream, "
                        "YouTube: https://youtube.com/watch?v=..., Webcam: 0, File: /path/to/video.mp4",
        examples=["rtsp://admin:pass@192.168.1.100:554/stream"]
    )
    camera_id: Optional[int] = Field(None, description="Camera identifier for MQTT routing")
    zone_id: Optional[int] = Field(None, description="Zone identifier for location reporting")
    model_name: str = Field("best", description="Detection model name")
    confidence: float = Field(0.4, description="PPE confidence threshold 0.1–0.95")
    detect_faces: bool = Field(False, description="Enable face recognition during streaming")


class ConvertRequest(BaseModel):
    """Configuration for annotation format conversion."""
    remap: dict = Field(
        {}, description="Class ID remapping: {old_id: new_id}. "
                        "Example: {0: 3} maps Roboflow class 0 to helmet (class 3).",
        examples=[{"0": 3, "1": 4}]
    )
    source_dir: str = Field(
        "raw_labels", description="Source directory name within the company's epi_check folder"
    )


class AnnotationSave(BaseModel):
    """Save YOLO bounding box annotations for an image."""
    image_filename: str = Field(
        ..., description="Image filename to save annotations for",
        examples=["photo_001.jpg"]
    )
    annotations: list[dict] = Field(
        [], description="List of YOLO annotations: [{class_id, cx, cy, w, h}]. "
                        "All coordinates are normalized 0.0–1.0 relative to image size.",
        examples=[[
            {"class_id": 3, "cx": 0.447, "cy": 0.333, "w": 0.067, "h": 0.046},
            {"class_id": 5, "cx": 0.500, "cy": 0.500, "w": 0.400, "h": 0.900},
        ]]
    )


class FaceRegisterRequest(BaseModel):
    """Register a person for face recognition."""
    person_code: str = Field(
        ..., max_length=50,
        description="Unique person identifier (lowercase, no spaces). Used as folder name.",
        examples=["carlos_santos"]
    )
    person_name: str = Field(
        ..., max_length=255,
        description="Full display name shown on detection results.",
        examples=["Carlos Santos"]
    )
    badge_id: str = Field(
        "", description="Employee badge or ID number (optional).",
        examples=["EMP001"]
    )
    image_base64: Optional[str] = Field(
        None, description="Base64-encoded face photo (alternative to file upload)"
    )


class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = Field(True, description="True if request was successful")
    message: str = Field("OK", description="Human-readable status message")
    data: Optional[dict | list] = Field(None, description="Response payload")
    company_id: Optional[int] = Field(None, description="Company ID that was used", examples=[1])
