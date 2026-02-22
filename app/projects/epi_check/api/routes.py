"""
EPI Check API v3 — REST endpoints for PPE detection, face recognition,
annotation converter, training, upload, and streaming.
Full OpenAPI documentation with summaries, descriptions, and examples.
"""
import base64
import json
import shutil
import time
import uuid
import glob
from pathlib import Path
from typing import Optional
from collections import defaultdict

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi import Path
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

from app.core.security import get_authenticated_company, get_ui_company
from app.core.company import CompanyData
from app.projects.epi_check.engine.detector import (
    epi_engine, ALL_PPE_CLASSES, ALL_CLASS_COLORS,
)
from app.projects.epi_check.models.schemas import (
    PPEConfig, DetectRequest, TrainRequest, VideoSource,
    ConvertRequest, AnnotationSave, FaceRegisterRequest, APIResponse,
)
from app.streaming.manager import stream_manager
from app.mqtt.client import mqtt_client

router = APIRouter()


# ======================================================================
# PPE CONFIGURATION
# ======================================================================
@router.get("/config",
    tags=["PPE Configuration"],
    summary="Get PPE Configuration",
    description="Returns the current PPE class configuration for a company, including which classes "
                "are enabled/disabled, active class IDs, and the full class registry.\n\n"
                "**Active classes** are used for training dataset generation, detection, and compliance checks. "
                "The `person` class is special — it's detected but never counted as a 'required' PPE.",
    response_description="PPE config with active_classes mapping and all_classes reference",
)
async def get_ppe_config(company_id: int = Depends(get_ui_company)):
    config = epi_engine.get_ppe_config(company_id)
    active = epi_engine.get_active_classes(company_id)
    return {"config": config, "active_classes": active, "all_classes": ALL_PPE_CLASSES}


@router.post("/config",
    tags=["PPE Configuration"],
    summary="Save PPE Configuration",
    description="Update which PPE classes are enabled for detection and compliance checking.\n\n"
                "**Important:** After changing this configuration:\n"
                "- Re-generate the dataset (POST `/dataset/generate`) to apply new class filtering\n"
                "- Re-train the model if classes were added or removed\n\n"
                "Only enable classes that you have annotated training images for.",
    response_description="Saved configuration",
)
async def save_ppe_config(config: PPEConfig, company_id: int = Depends(get_ui_company)):
    epi_engine.save_ppe_config(company_id, config.model_dump())
    return {"success": True, "config": config.model_dump()}


# ======================================================================
# PHOTO UPLOAD (per category)
# ======================================================================
@router.post("/upload/photos",
    tags=["Photo Upload"],
    summary="Upload Training Photos by Category",
    description="Upload one or more training images organized by PPE category. "
                "Images are validated and stored in the `photos_raw/{category}/` folder.\n\n"
                "**Categories:** `helmet`, `boots`, `person`, `gloves`, `thermal_coat`, "
                "`thermal_pants`, `full_body`\n\n"
                "**Supported formats:** JPEG, PNG, BMP, WebP\n\n"
                "These photos are used as raw material for the visual annotation tool. "
                "After upload, annotate them via the `/annotate/` endpoints.",
    response_description="Upload results per file with dimensions",
)
async def upload_photos(
    category: str = Form(..., description="PPE category name", examples=["helmet"]),
    files: list[UploadFile] = File(..., description="Image files to upload"),
    company_id: int = Depends(get_ui_company),
):
    dest = CompanyData.epi(company_id, "photos_raw", category)
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        data = await f.read()
        fpath = dest / f.filename
        fpath.write_bytes(data)
        img = cv2.imread(str(fpath))
        if img is not None:
            h, w = img.shape[:2]
            results.append({"file": f.filename, "ok": True, "size": f"{w}x{h}"})
        else:
            fpath.unlink()
            results.append({"file": f.filename, "ok": False, "error": "Not a valid image"})
    return {"category": category, "uploaded": len([r for r in results if r["ok"]]), "results": results}


# ======================================================================
# BULK UPLOAD (images + YOLO TXT labels)
# ======================================================================
@router.post("/upload/bulk",
    tags=["Photo Upload"],
    summary="Bulk Upload Images + YOLO TXT Labels",
    description="Upload pre-annotated images paired with YOLO TXT label files. "
                "Files are matched by base filename (e.g., `photo_001.jpg` + `photo_001.txt`).\n\n"
                "**YOLO TXT format** (5 values per line):\n"
                "```\nclass_id  center_x  center_y  width  height\n"
                "3 0.447115 0.332933 0.067308 0.045673\n```\n\n"
                "**Class ID Remap:** If your source dataset uses different class IDs, provide a JSON mapping.\n"
                "Example: `{\"0\": 3}` maps source class 0 → helmet (class 3).\n\n"
                "Only annotations matching **active** PPE classes are kept. Configure classes via POST `/config` first.",
    response_description="Count of paired (image+label) and image-only files",
)
async def upload_bulk(
    category: str = Form("mixed", description="Category label", examples=["mixed"]),
    remap_json: str = Form("{}", description='Class ID remap as JSON string, e.g. {"0":3,"1":4}'),
    files: list[UploadFile] = File(..., description="Image files (.jpg/.png) and YOLO label files (.txt)"),
    company_id: int = Depends(get_ui_company),
):
    ann_dir = CompanyData.epi(company_id, "annotations")
    ann_dir.mkdir(parents=True, exist_ok=True)
    remap = {}
    try:
        raw = json.loads(remap_json)
        remap = {int(k): int(v) for k, v in raw.items()} if raw else {}
    except Exception:
        pass

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images_data = {}
    labels_data = {}

    for f in files:
        data = await f.read()
        stem = Path(f.filename).stem
        ext = Path(f.filename).suffix.lower()
        if ext in img_exts:
            images_data[stem] = (f.filename, data)
        elif ext == ".txt":
            labels_data[stem] = (f.filename, data)

    active = epi_engine.get_active_classes(company_id)
    active_ids = set(active.keys())
    paired = 0
    images_only = 0

    for stem, (img_name, img_bytes) in images_data.items():
        (ann_dir / img_name).write_bytes(img_bytes)
        if stem in labels_data:
            lbl_name, lbl_bytes = labels_data[stem]
            lbl_text = lbl_bytes.decode("utf-8", errors="ignore")
            filtered = []
            for line in lbl_text.strip().split("\n"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_id = int(parts[0])
                    new_id = remap.get(old_id, old_id)
                    if new_id in active_ids:
                        parts[0] = str(new_id)
                        filtered.append(" ".join(parts))
            (ann_dir / (stem + ".txt")).write_text("\n".join(filtered))
            paired += 1
        else:
            images_only += 1
    return {"paired": paired, "images_only": images_only, "total_files": len(files)}


# ======================================================================
# ANNOTATION CONVERTER (polygon/OBB → YOLO bbox)
# ======================================================================
@router.post("/convert/upload",
    tags=["Annotation Converter"],
    summary="Upload & Convert Polygon/OBB → YOLO BBox",
    description="Upload Roboflow polygon or OBB annotation files and convert them to standard YOLO bbox format.\n\n"
                "**Input format (9 values — polygon/OBB from Roboflow):**\n"
                "```\n0  x1 y1  x2 y2  x3 y3  x4 y4\n```\n\n"
                "**Output format (5 values — YOLO bbox):**\n"
                "```\n3  center_x  center_y  width  height\n```\n\n"
                "Images and TXT files are uploaded together. Converted labels are saved to `annotations/` folder. "
                "The converter automatically detects the format (5-value = already YOLO, 9-value = polygon/OBB).\n\n"
                "**Class ID Remap:** Use `{\"0\": 3}` to map Roboflow class 0 → helmet (class 3).",
    response_description="Number of files and bounding boxes converted",
)
async def convert_upload(
    remap_json: str = Form("{}", description='Class ID remap JSON, e.g. {"0":3}'),
    files: list[UploadFile] = File(..., description="Image + annotation TXT files from Roboflow export"),
    company_id: int = Depends(get_ui_company),
):
    raw_dir = CompanyData.epi(company_id, "raw_labels")
    raw_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = CompanyData.epi(company_id, "annotations")

    for f in files:
        data = await f.read()
        (raw_dir / f.filename).write_bytes(data)

    remap = {}
    try:
        raw = json.loads(remap_json)
        remap = {int(k): int(v) for k, v in raw.items()} if raw else {}
    except Exception:
        pass

    result = epi_engine.converter.convert_directory(
        str(raw_dir), str(ann_dir), remap=remap, copy_images=True,
    )
    return {"files_converted": result["files"], "boxes_converted": result["boxes"],
            "remap": remap, "output": str(ann_dir)}


@router.post("/convert/run",
    tags=["Annotation Converter"],
    summary="Convert Existing raw_labels/ Files",
    description="Convert annotation files already present in the `raw_labels/` folder "
                "(previously uploaded or placed manually via SSH/drive).\n\n"
                "Useful when files were uploaded outside the API or need re-conversion with different remap settings.",
    response_description="Conversion statistics",
)
async def convert_existing(
    remap_json: str = Form("{}", description='Class ID remap JSON'),
    company_id: int = Depends(get_ui_company),
):
    raw_dir = CompanyData.epi(company_id, "raw_labels")
    ann_dir = CompanyData.epi(company_id, "annotations")
    remap = {}
    try:
        raw = json.loads(remap_json)
        remap = {int(k): int(v) for k, v in raw.items()} if raw else {}
    except Exception:
        pass
    result = epi_engine.converter.convert_directory(
        str(raw_dir), str(ann_dir), remap=remap, copy_images=True,
    )
    return {"files_converted": result["files"], "boxes_converted": result["boxes"]}


# ======================================================================
# VISUAL ANNOTATION
# ======================================================================
@router.get("/annotate/images",
    tags=["Visual Annotation"],
    summary="List Images Available for Annotation",
    description="Returns a deduplicated list of all images from `annotations/` and `photos_raw/` folders. "
                "Each entry indicates whether a matching `.txt` label file exists (`has_label: true`).",
    response_description="List of images with annotation status",
)
async def list_annotation_images(company_id: int = Depends(get_ui_company)):
    ann_dir = CompanyData.epi(company_id, "annotations")
    raw_dir = CompanyData.epi(company_id, "photos_raw")
    images = []
    for d in [ann_dir, raw_dir]:
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for f in d.rglob(ext):
                images.append({"name": f.name, "path": str(f),
                               "has_label": (ann_dir / (f.stem + ".txt")).exists()})
    seen = set()
    unique = []
    for img in images:
        if img["name"] not in seen:
            seen.add(img["name"])
            unique.append(img)
    return unique


@router.get("/annotate/image/{filename}",
    tags=["Visual Annotation"],
    summary="Serve Image for Annotation",
    description="Returns the raw image file for display in the browser-based annotation tool. "
                "Searches `annotations/` first, then `photos_raw/` subfolders.",
    response_description="JPEG image binary",
)
async def get_annotation_image(
    filename: str = Path(..., description="Image filename, e.g. photo_001.jpg"),
    company_id: int = Depends(get_ui_company),
):
    for d in [CompanyData.epi(company_id, "annotations"),
              CompanyData.epi(company_id, "photos_raw")]:
        for ext_d in [d] + list(d.iterdir()) if d.is_dir() else [d]:
            fp = ext_d / filename if ext_d.is_dir() else ext_d
            if fp.exists() and fp.name == filename:
                return StreamingResponse(open(fp, "rb"), media_type="image/jpeg")
    raise HTTPException(404, "Image not found")


@router.get("/annotate/labels/{filename}",
    tags=["Visual Annotation"],
    summary="Get YOLO Labels for an Image",
    description="Returns existing YOLO bounding box annotations for an image. "
                "Each label has: `class_id`, `cx` (center x), `cy` (center y), "
                "`w` (width), `h` (height) — all normalized 0.0–1.0.",
    response_description="List of YOLO labels",
)
async def get_labels(
    filename: str = Path(..., description="Image filename"),
    company_id: int = Depends(get_ui_company),
):
    stem = Path(filename).stem
    lbl_path = CompanyData.epi(company_id, "annotations", stem + ".txt")
    if not lbl_path.exists():
        return {"labels": []}
    labels = []
    for line in lbl_path.read_text().strip().split("\n"):
        parts = line.strip().split()
        if len(parts) == 5:
            labels.append({
                "class_id": int(parts[0]),
                "cx": float(parts[1]), "cy": float(parts[2]),
                "w": float(parts[3]), "h": float(parts[4]),
            })
    return {"labels": labels}


@router.post("/annotate/save",
    tags=["Visual Annotation"],
    summary="Save Drawn Annotations",
    description="Save YOLO bounding box annotations created in the browser annotation tool. "
                "Annotations are written as a `.txt` file alongside the image in `annotations/`.\n\n"
                "**Format per annotation:** `{class_id} {center_x} {center_y} {width} {height}` "
                "(all values normalized 0.0–1.0).\n\n"
                "If the image is in `photos_raw/` but not yet in `annotations/`, it is copied automatically.",
    response_description="Number of annotations saved",
)
async def save_annotations(data: AnnotationSave, company_id: int = Depends(get_ui_company)):
    ann_dir = CompanyData.epi(company_id, "annotations")
    stem = Path(data.image_filename).stem
    lbl_path = ann_dir / (stem + ".txt")

    img_dest = ann_dir / data.image_filename
    if not img_dest.exists():
        for d in [CompanyData.epi(company_id, "photos_raw")]:
            for src in d.rglob(data.image_filename):
                shutil.copy2(str(src), str(img_dest))
                break

    lines = []
    for ann in data.annotations:
        line = f"{ann['class_id']} {ann['cx']:.6f} {ann['cy']:.6f} {ann['w']:.6f} {ann['h']:.6f}"
        lines.append(line)
    lbl_path.write_text("\n".join(lines))
    return {"saved": len(lines), "file": str(lbl_path)}


# ======================================================================
# ANNOTATION STATUS
# ======================================================================
@router.get("/annotations/status",
    tags=["Visual Annotation"],
    summary="Annotation Statistics",
    description="Returns how many images have annotations and the class distribution across all labels.",
    response_description="Annotation counts and class distribution",
)
async def annotation_status(company_id: int = Depends(get_ui_company)):
    ann_dir = CompanyData.epi(company_id, "annotations")
    labels = list(ann_dir.glob("*.txt"))
    images = list(ann_dir.glob("*.jpg")) + list(ann_dir.glob("*.jpeg")) + list(ann_dir.glob("*.png"))
    class_counts = defaultdict(int)
    for lbl in labels:
        for line in lbl.read_text().strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                name = ALL_PPE_CLASSES.get(cid, f"class_{cid}")
                class_counts[name] += 1
    return {"annotated_images": len(labels), "total_images": len(images), "class_counts": dict(class_counts)}


@router.get("/photos/summary",
    tags=["Photo Upload"],
    summary="Photo Counts per Category",
    description="Returns the number of training photos uploaded per PPE category in `photos_raw/`.",
    response_description="Photo count per category",
)
async def photo_summary(company_id: int = Depends(get_ui_company)):
    raw_dir = CompanyData.epi(company_id, "photos_raw")
    summary = {}
    for cls_name in list(ALL_PPE_CLASSES.values()) + ["full_body"]:
        d = raw_dir / cls_name
        n = len(list(d.glob("*.*"))) if d.exists() else 0
        summary[cls_name] = n
    return summary


# ======================================================================
# DATASET GENERATION
# ======================================================================
@router.post("/dataset/generate",
    tags=["Dataset & Training"],
    summary="Generate YOLOv8 Train/Valid Dataset",
    description="Creates a YOLOv8 dataset from annotated images in `annotations/` folder.\n\n"
                "**Process:**\n"
                "1. Finds all image+label pairs in `annotations/`\n"
                "2. Filters to only **active** PPE classes\n"
                "3. Remaps class IDs to sequential 0-based (required by YOLO)\n"
                "4. Splits into train/valid sets\n"
                "5. Generates `data.yaml` configuration file\n\n"
                "**Prerequisites:** Upload and annotate images first. At least 50 images per class recommended.",
    response_description="Dataset split statistics and class mapping",
)
async def generate_dataset(
    train_split: float = Form(0.8, description="Train/valid split ratio (0.5–0.95)", examples=[0.8]),
    company_id: int = Depends(get_ui_company),
):
    result = epi_engine.generate_dataset(company_id, train_split)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.get("/dataset/stats",
    tags=["Dataset & Training"],
    summary="Dataset Image Counts",
    description="Returns the current number of images in the train and valid dataset folders.",
    response_description="Train and valid image counts",
)
async def dataset_stats(company_id: int = Depends(get_ui_company)):
    train_dir = CompanyData.epi(company_id, "dataset", "train", "images")
    valid_dir = CompanyData.epi(company_id, "dataset", "valid", "images")
    return {
        "train_images": len(list(train_dir.glob("*.*"))) if train_dir.exists() else 0,
        "valid_images": len(list(valid_dir.glob("*.*"))) if valid_dir.exists() else 0,
    }


# ======================================================================
# TRAINING
# ======================================================================
@router.post("/train/start",
    tags=["Dataset & Training"],
    summary="Start Model Training",
    description="Launch a YOLOv8 training run in the background. Non-blocking — returns immediately.\n\n"
                "**Requirements:**\n"
                "- Dataset must be generated first (POST `/dataset/generate`)\n"
                "- GPU strongly recommended (T4: ~20-30 min, CPU: ~15-20 hours)\n\n"
                "**Model sizes:**\n"
                "- `yolov8n.pt` — Nano (fastest, less accurate)\n"
                "- `yolov8s.pt` — Small\n"
                "- `yolov8m.pt` — Medium (recommended balance)\n"
                "- `yolov8l.pt` — Large (more accurate, slower)\n"
                "- `yolov8x.pt` — Extra Large (most accurate, slowest)\n\n"
                "Poll GET `/train/status` for progress.",
    response_description="Training status (preparing/training/complete/error)",
)
async def start_training(req: TrainRequest, company_id: int = Depends(get_ui_company)):
    return epi_engine.train_model(company_id, req.base_model, req.epochs,
                                   req.batch_size, req.img_size, req.patience)


@router.get("/train/status",
    tags=["Dataset & Training"],
    summary="Poll Training Progress",
    description="Check the current status of a training run.\n\n"
                "**Status values:**\n"
                "- `idle` — No training started\n"
                "- `preparing` — Setting up training\n"
                "- `training` — In progress (includes epoch count)\n"
                "- `complete` — Finished (includes model_path)\n"
                "- `error` — Failed (includes error message)\n\n"
                "Poll every 5 seconds during training.",
    response_description="Training status with epoch progress or error details",
)
async def train_status(company_id: int = Depends(get_ui_company)):
    return epi_engine.get_train_status(company_id)


# ======================================================================
# DETECTION — Image
# ======================================================================
@router.post("/detect/image",
    tags=["Detection — Image"],
    summary="Detect PPE from Base64 Image (REST API)",
    description="Run PPE detection on a base64-encoded image. Designed for **REST API integration** "
                "(Node-RED, edge devices, external systems).\n\n"
                "**Features:**\n"
                "- PPE detection with configurable confidence threshold\n"
                "- Optional face recognition (set `detect_faces: true`)\n"
                "- Adjustable face similarity threshold\n"
                "- Camera/zone tracking for MQTT alert routing\n\n"
                "**MQTT Alert:** When `compliant: false`, an alert is automatically published to "
                "`smartx/vision/{company_id}/alert` with missing items and person identity.",
    response_model=APIResponse,
    response_description="Detection result with compliance status, detections, and face matches",
)
async def detect_image_b64(req: DetectRequest, company_id: int = Depends(get_ui_company)):
    if not req.image_base64:
        raise HTTPException(400, "image_base64 required")
    raw = base64.b64decode(req.image_base64.split(",")[-1])
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")
    result = epi_engine.detect_image(company_id, img, req.model_name,
                                      req.confidence, req.detect_faces,
                                      req.face_threshold)
    if not result["compliant"]:
        await mqtt_client.publish_alert(company_id, "EPI_NON_COMPLIANT", {
            "missing": result["missing"], "camera_id": req.camera_id,
            "faces": result.get("faces", []),
        })
    return APIResponse(data=result, company_id=company_id)


@router.post("/detect/upload",
    tags=["Detection — Image"],
    summary="Detect PPE from Uploaded Image (UI)",
    description="Upload an image file and run PPE detection + optional face recognition.\n\n"
                "Returns the **annotated image** (base64) with bounding boxes, compliance banner, "
                "and person names drawn on the image. Also saves a snapshot to `results/` folder.\n\n"
                "**Thresholds:**\n"
                "- `confidence` — PPE detection threshold (0.1–0.9). Recommended: 0.45 balanced, 0.60+ to reduce false positives\n"
                "- `face_threshold` — Face similarity threshold (0.2–0.8). Lower = more lenient, higher = stricter matching",
    response_description="Detection result with annotated base64 image and snapshot URL",
)
async def detect_image_upload(
    file: UploadFile = File(..., description="Image file (JPEG, PNG)"),
    model_name: str = Form("best", description="Model name"),
    confidence: float = Form(0.4, description="PPE confidence threshold 0.1–0.9"),
    detect_faces: bool = Form(False, description="Enable face recognition"),
    face_threshold: float = Form(0.45, description="Face similarity threshold 0.2–0.8"),
    company_id: int = Depends(get_ui_company),
):
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")
    annotated, result = epi_engine.detect_and_annotate(
        company_id, img, model_name, confidence, detect_faces, face_threshold,
    )
    snap_dir = CompanyData.epi(company_id, "results")
    snap_name = f"result_{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(str(snap_dir / snap_name), annotated)
    result["snapshot_url"] = f"/api/v1/epi/results/{snap_name}?company_id={company_id}"
    _, buf = cv2.imencode(".jpg", annotated)
    result["annotated_base64"] = base64.b64encode(buf).decode()
    return result


# ======================================================================
# DETECTION — Video
# ======================================================================
@router.post("/detect/video",
    tags=["Detection — Video"],
    summary="Process Uploaded Video File",
    description="Upload a video file (MP4, AVI, MOV) and process it frame-by-frame for PPE compliance.\n\n"
                "**Parameters:**\n"
                "- `skip_frames` — Process every Nth frame (higher = faster but less coverage)\n"
                "- Results include per-frame details (first 50) and overall compliance rate\n\n"
                "**Performance:** A 1-minute video at 30fps with skip_frames=5 processes ~360 frames.",
    response_description="Compliance rate and per-frame detection details",
)
async def detect_video_upload(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV)"),
    model_name: str = Form("best", description="Model name"),
    confidence: float = Form(0.4, description="PPE confidence threshold"),
    skip_frames: int = Form(5, description="Process every Nth frame (1=all, 5=every 5th)"),
    detect_faces: bool = Form(False, description="Enable face recognition"),
    company_id: int = Depends(get_ui_company),
):
    temp_dir = CompanyData.epi(company_id, "temp")
    temp_path = temp_dir / f"video_{uuid.uuid4().hex[:8]}{Path(file.filename).suffix}"
    temp_path.write_bytes(await file.read())
    results = epi_engine.process_video(company_id, str(temp_path), model_name,
                                        confidence, skip_frames, detect_faces=detect_faces)
    compliant_count = sum(1 for r in results if r["compliant"])
    total = len(results)
    return {"frames_processed": total, "compliant_frames": compliant_count,
            "non_compliant_frames": total - compliant_count,
            "compliance_rate": round(compliant_count / max(total, 1) * 100, 1),
            "details": results[:50]}


@router.post("/detect/youtube",
    tags=["Detection — Video"],
    summary="Process YouTube Video",
    description="Download a YouTube video and process it frame-by-frame for PPE compliance.\n\n"
                "**Note:** Requires `yt-dlp` package. Video is downloaded at max 720p resolution. "
                "Temporary file is deleted after processing.\n\n"
                "`max_frames` limits the total frames processed to control execution time.",
    response_description="Compliance rate and per-frame details",
)
async def detect_youtube(
    url: str = Form(..., description="YouTube URL", examples=["https://www.youtube.com/watch?v=abc123"]),
    model_name: str = Form("best", description="Model name"),
    confidence: float = Form(0.4, description="PPE confidence threshold"),
    max_frames: int = Form(100, description="Maximum frames to process"),
    detect_faces: bool = Form(False, description="Enable face recognition"),
    company_id: int = Depends(get_ui_company),
):
    import yt_dlp
    temp_dir = CompanyData.epi(company_id, "temp")
    temp_path = temp_dir / f"yt_{uuid.uuid4().hex[:8]}.mp4"
    ydl_opts = {"format": "best[height<=720]", "outtmpl": str(temp_path), "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    results = epi_engine.process_video(company_id, str(temp_path), model_name,
                                        confidence, skip_frames=10, max_frames=max_frames,
                                        detect_faces=detect_faces)
    compliant_count = sum(1 for r in results if r["compliant"])
    total = len(results)
    temp_path.unlink(missing_ok=True)
    return {"source": url, "frames_processed": total,
            "compliant_frames": compliant_count,
            "compliance_rate": round(compliant_count / max(total, 1) * 100, 1),
            "details": results[:50]}


# ======================================================================
# FACE RECOGNITION
# ======================================================================
@router.post("/faces/register",
    tags=["Face Recognition"],
    summary="Register Person with Face Photo",
    description="Register a new person for face recognition by uploading a face photo.\n\n"
                "**Tips for best accuracy:**\n"
                "- Upload **3–5 photos** per person (call this endpoint multiple times with same `person_code`)\n"
                "- Use different **angles** (front, slight left, slight right)\n"
                "- Use different **lighting** conditions\n"
                "- Face should be clearly visible, not obscured by PPE\n\n"
                "The face database is rebuilt automatically after each registration. "
                "Multiple embeddings per person are averaged for more robust matching.",
    response_description="Registration result with photo count",
)
async def register_face(
    person_code: str = Form(..., description="Unique person ID (lowercase, no spaces)", examples=["carlos_santos"]),
    person_name: str = Form(..., description="Full display name", examples=["Carlos Santos"]),
    badge_id: str = Form("", description="Employee badge number", examples=["EMP001"]),
    file: UploadFile = File(..., description="Face photo (JPEG, PNG)"),
    company_id: int = Depends(get_ui_company),
):
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")
    result = epi_engine.face_engine.register_face(
        company_id, person_code, person_name, badge_id, img,
    )
    return {"success": True, **result}


@router.get("/faces/people",
    tags=["Face Recognition"],
    summary="List Registered People",
    description="Returns all registered people for a company with their photo counts. "
                "People with more photos generally have more accurate face matching.",
    response_description="List of registered people with metadata",
)
async def list_people(company_id: int = Depends(get_ui_company)):
    return epi_engine.face_engine.list_people(company_id)


@router.post("/faces/rebuild",
    tags=["Face Recognition"],
    summary="Rebuild Face Embeddings Database",
    description="Re-process all registered face photos and rebuild the embedding database. "
                "Run this after manually adding/removing photos from the `people/` folder, "
                "or if face recognition accuracy seems degraded.\n\n"
                "**Note:** This is done automatically after each `/faces/register` call.",
    response_description="Number of people loaded into face DB",
)
async def rebuild_face_db(company_id: int = Depends(get_ui_company)):
    db = epi_engine.face_engine.build_face_db(company_id)
    return {"people_loaded": len(db)}


# ======================================================================
# LIVE STREAMING
# ======================================================================
@router.post("/stream/start",
    tags=["Live Streaming"],
    summary="Start Live Stream with Real-Time Detection",
    description="Start a live video stream with real-time PPE detection and optional face recognition.\n\n"
                "**Source types:**\n"
                "- `rtsp` — IP camera: `rtsp://admin:pass@192.168.1.100:554/stream`\n"
                "- `youtube` — YouTube live or video: `https://youtube.com/watch?v=...`\n"
                "- `webcam` — Local webcam: `0` (device index)\n"
                "- `file` — Video file path: `/path/to/video.mp4`\n\n"
                "After starting, use GET `/stream/{session_id}/feed` to display the MJPEG feed in a browser.",
    response_description="Stream session ID and source info",
)
async def start_stream(
    source: str = Form(..., description="Stream URL or device", examples=["rtsp://admin:pass@192.168.1.100:554/stream"]),
    source_type: str = Form("rtsp", description="Source type: rtsp, youtube, webcam, file"),
    model_name: str = Form("best", description="Detection model name"),
    confidence: float = Form(0.4, description="PPE confidence threshold"),
    detect_faces: bool = Form(False, description="Enable face recognition"),
    face_threshold: float = Form(0.45, description="Face similarity threshold"),
    company_id: int = Depends(get_ui_company),
):
    def process_fn(frame):
        return epi_engine.detect_and_annotate(company_id, frame, model_name, confidence, detect_faces, face_threshold)
    session = stream_manager.create_session(source, source_type, company_id, process_fn)
    session.start()
    return {"session_id": session.session_id, "source": source, "source_type": source_type}


@router.get("/stream/{session_id}/feed",
    tags=["Live Streaming"],
    summary="MJPEG Video Feed",
    description="Returns a continuous MJPEG stream for browser display.\n\n"
                "**Usage in HTML:**\n"
                "```html\n<img src=\"/api/v1/epi/stream/{session_id}/feed\" />\n```\n\n"
                "The feed includes annotated frames with PPE bounding boxes, face labels, "
                "and compliance status banner.",
    response_description="Continuous MJPEG stream (multipart/x-mixed-replace)",
)
async def stream_feed(session_id: str):
    session = stream_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, "Stream not found")

    def generate():
        while session._running:
            jpeg = stream_manager.get_frame_jpeg(session_id, quality=70)
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/stream/{session_id}/status",
    tags=["Live Streaming"],
    summary="Stream Session Status",
    description="Get real-time info about a running stream session including FPS, frame count, "
                "and the latest PPE detection result (compliance status, missing items, recognized faces).",
    response_description="Stream info with latest detection result",
)
async def stream_status(session_id: str):
    session = stream_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, "Stream not found")
    return {**session.info, "latest_result": session.latest_result}


@router.post("/stream/{session_id}/stop",
    tags=["Live Streaming"],
    summary="Stop Stream",
    description="Stop a running stream session and release the video capture resources.",
    response_description="Confirmation",
)
async def stop_stream(session_id: str):
    stream_manager.stop_session(session_id)
    return {"stopped": True}


@router.get("/stream/sessions",
    tags=["Live Streaming"],
    summary="List Active Stream Sessions",
    description="Returns all active stream sessions for the specified company, "
                "including source info, FPS, and frame counts.",
    response_description="List of active stream sessions",
)
async def list_streams(company_id: int = Depends(get_ui_company)):
    return [s for s in stream_manager.list_sessions() if s["company_id"] == company_id]


# ======================================================================
# MODELS / RESULTS / STATS
# ======================================================================
@router.get("/models",
    tags=["Models & Results"],
    summary="List Trained Models",
    description="Returns all `.pt` model files available for the company, "
                "including file size. Use the model `name` in detection endpoints.",
    response_description="List of models with paths and sizes",
)
async def list_models(company_id: int = Depends(get_ui_company)):
    return epi_engine.list_models(company_id)


@router.get("/results/{filename}",
    tags=["Models & Results"],
    summary="Serve Detection Result Snapshot",
    description="Returns an annotated result image saved during detection. "
                "The `snapshot_url` field in detection responses provides the full path.",
    response_description="JPEG image binary",
)
async def get_result_image(filename: str, company_id: int = Depends(get_ui_company)):
    fpath = CompanyData.epi(company_id, "results", filename)
    if not fpath.exists():
        raise HTTPException(404, "File not found")
    return StreamingResponse(open(fpath, "rb"), media_type="image/jpeg")


@router.get("/stats",
    tags=["Models & Results"],
    summary="System Statistics",
    description="Comprehensive system statistics for a company including PPE configuration, "
                "active classes, model count, registered people count, disk usage, and training status.",
    response_description="Full system stats",
)
async def epi_stats(company_id: int = Depends(get_ui_company)):
    config = epi_engine.get_ppe_config(company_id)
    active = epi_engine.get_active_classes(company_id)
    models = epi_engine.list_models(company_id)
    storage = CompanyData.disk_usage(company_id)
    people = epi_engine.face_engine.list_people(company_id)
    return {"company_id": company_id, "ppe_config": config, "active_classes": active,
            "models_count": len(models), "people_count": len(people),
            "storage": storage, "train_status": epi_engine.get_train_status(company_id)}
