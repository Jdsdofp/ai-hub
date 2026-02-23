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
@router.get("/config", tags=["PPE Configuration"], summary="Get PPE Configuration")
async def get_ppe_config(company_id: int = Depends(get_ui_company)):
    try:
        config = epi_engine.get_ppe_config(company_id)
        active = epi_engine.get_active_classes(company_id)
        return {"config": config, "active_classes": active, "all_classes": ALL_PPE_CLASSES}
    except Exception as e:
        logger.error(f"[Company {company_id}] get_ppe_config error: {e}")
        raise HTTPException(500, detail=f"Failed to get PPE config: {str(e)}")


@router.post("/config", tags=["PPE Configuration"], summary="Save PPE Configuration")
async def save_ppe_config(config: PPEConfig, company_id: int = Depends(get_ui_company)):
    try:
        epi_engine.save_ppe_config(company_id, config.model_dump())
        return {"success": True, "config": config.model_dump()}
    except Exception as e:
        logger.error(f"[Company {company_id}] save_ppe_config error: {e}")
        raise HTTPException(500, detail=f"Failed to save PPE config: {str(e)}")


# ======================================================================
# PHOTO UPLOAD
# ======================================================================
@router.post("/upload/photos", tags=["Photo Upload"], summary="Upload Training Photos by Category")
async def upload_photos(
    category: str = Form(...),
    files: list[UploadFile] = File(...),
    company_id: int = Depends(get_ui_company),
):
    try:
        dest = CompanyData.epi(company_id, "photos_raw", category)
        dest.mkdir(parents=True, exist_ok=True)
        results = []
        for f in files:
            try:
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
            except Exception as fe:
                results.append({"file": f.filename, "ok": False, "error": str(fe)})
        return {"category": category, "uploaded": len([r for r in results if r["ok"]]), "results": results}
    except Exception as e:
        logger.error(f"[Company {company_id}] upload_photos error: {e}")
        raise HTTPException(500, detail=f"Upload failed: {str(e)}")


# ======================================================================
# BULK UPLOAD
# ======================================================================
@router.post("/upload/bulk", tags=["Photo Upload"], summary="Bulk Upload Images + YOLO TXT Labels")
async def upload_bulk(
    category: str = Form("mixed"),
    remap_json: str = Form("{}"),
    files: list[UploadFile] = File(...),
    company_id: int = Depends(get_ui_company),
):
    try:
        ann_dir = CompanyData.epi(company_id, "annotations")
        ann_dir.mkdir(parents=True, exist_ok=True)
        remap = {}
        try:
            raw = json.loads(remap_json)
            remap = {int(k): int(v) for k, v in raw.items()} if raw else {}
        except Exception as je:
            raise HTTPException(400, detail=f"Invalid remap_json: {str(je)}")

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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] upload_bulk error: {e}")
        raise HTTPException(500, detail=f"Bulk upload failed: {str(e)}")


# ======================================================================
# ANNOTATION CONVERTER
# ======================================================================
@router.post("/convert/upload", tags=["Annotation Converter"], summary="Upload & Convert Polygon/OBB → YOLO BBox")
async def convert_upload(
    remap_json: str = Form("{}"),
    files: list[UploadFile] = File(...),
    company_id: int = Depends(get_ui_company),
):
    try:
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
        except Exception as je:
            raise HTTPException(400, detail=f"Invalid remap_json: {str(je)}")

        result = epi_engine.converter.convert_directory(
            str(raw_dir), str(ann_dir), remap=remap, copy_images=True,
        )
        return {"files_converted": result["files"], "boxes_converted": result["boxes"],
                "remap": remap, "output": str(ann_dir)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] convert_upload error: {e}")
        raise HTTPException(500, detail=f"Conversion failed: {str(e)}")


@router.post("/convert/run", tags=["Annotation Converter"], summary="Convert Existing raw_labels/ Files")
async def convert_existing(
    remap_json: str = Form("{}"),
    company_id: int = Depends(get_ui_company),
):
    try:
        raw_dir = CompanyData.epi(company_id, "raw_labels")
        ann_dir = CompanyData.epi(company_id, "annotations")
        remap = {}
        try:
            raw = json.loads(remap_json)
            remap = {int(k): int(v) for k, v in raw.items()} if raw else {}
        except Exception as je:
            raise HTTPException(400, detail=f"Invalid remap_json: {str(je)}")
        result = epi_engine.converter.convert_directory(
            str(raw_dir), str(ann_dir), remap=remap, copy_images=True,
        )
        return {"files_converted": result["files"], "boxes_converted": result["boxes"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] convert_existing error: {e}")
        raise HTTPException(500, detail=f"Conversion failed: {str(e)}")


# ======================================================================
# VISUAL ANNOTATION
# ======================================================================
@router.get("/annotate/images", tags=["Visual Annotation"], summary="List Images Available for Annotation")
async def list_annotation_images(company_id: int = Depends(get_ui_company)):
    try:
        ann_dir = CompanyData.epi(company_id, "annotations")
        raw_dir = CompanyData.epi(company_id, "photos_raw")
        images = []
        for d in [ann_dir, raw_dir]:
            if not d.exists():
                continue
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
    except Exception as e:
        logger.error(f"[Company {company_id}] list_annotation_images error: {e}")
        raise HTTPException(500, detail=f"Error listing images: {str(e)}")


@router.get("/annotate/image/{filename}", tags=["Visual Annotation"], summary="Serve Image for Annotation")
async def get_annotation_image(filename: str, company_id: int = Depends(get_ui_company)):
    try:
        search_dirs = [CompanyData.epi(company_id, "annotations")]
        raw_dir = CompanyData.epi(company_id, "photos_raw")
        search_dirs.append(raw_dir)
        if raw_dir.is_dir():
            for sub in raw_dir.iterdir():
                if sub.is_dir():
                    search_dirs.append(sub)
        for d in search_dirs:
            if not d.is_dir():
                continue
            fp = d / filename
            if fp.exists() and fp.is_file():
                return StreamingResponse(open(fp, "rb"), media_type="image/jpeg")
        raise HTTPException(404, detail=f"Image not found: {filename}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] get_annotation_image error: {e}")
        raise HTTPException(500, detail=f"Error loading image: {str(e)}")


@router.get("/annotate/labels/{filename}", tags=["Visual Annotation"], summary="Get YOLO Labels for an Image")
async def get_labels(filename: str, company_id: int = Depends(get_ui_company)):
    try:
        stem = Path(filename).stem
        lbl_path = CompanyData.epi(company_id, "annotations", stem + ".txt")
        if not lbl_path.exists():
            return {"labels": []}
        text = lbl_path.read_text().strip()
        if not text:
            return {"labels": []}
        labels = []
        for line in text.split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append({
                    "class_id": int(parts[0]),
                    "cx": float(parts[1]), "cy": float(parts[2]),
                    "w": float(parts[3]), "h": float(parts[4]),
                })
        return {"labels": labels}
    except Exception as e:
        logger.error(f"[Company {company_id}] get_labels error for {filename}: {e}")
        raise HTTPException(500, detail=f"Error reading labels: {str(e)}")


@router.post("/annotate/save", tags=["Visual Annotation"], summary="Save Drawn Annotations")
async def save_annotations(data: AnnotationSave, company_id: int = Depends(get_ui_company)):
    try:
        ann_dir = CompanyData.epi(company_id, "annotations")
        stem = Path(data.image_filename).stem
        lbl_path = ann_dir / (stem + ".txt")
        img_dest = ann_dir / data.image_filename
        if not img_dest.exists():
            raw_dir = CompanyData.epi(company_id, "photos_raw")
            for src in raw_dir.rglob(data.image_filename):
                shutil.copy2(str(src), str(img_dest))
                break
        lines = []
        for ann in data.annotations:
            cid = int(ann.get('class_id', ann.get('classId', 0)))
            cx = float(ann.get('cx', 0))
            cy = float(ann.get('cy', 0))
            w = float(ann.get('w', 0))
            h = float(ann.get('h', 0))
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        lbl_path.write_text("\n".join(lines))
        return {"saved": len(lines), "file": str(lbl_path)}
    except Exception as e:
        logger.error(f"[Company {company_id}] save_annotations error: {e}")
        raise HTTPException(500, detail=f"Error saving annotations: {str(e)}")


# ======================================================================
# ANNOTATION STATUS
# ======================================================================
@router.get("/annotations/status", tags=["Visual Annotation"], summary="Annotation Statistics")
async def annotation_status(company_id: int = Depends(get_ui_company)):
    try:
        ann_dir = CompanyData.epi(company_id, "annotations")
        labels = list(ann_dir.glob("*.txt"))
        images = list(ann_dir.glob("*.jpg")) + list(ann_dir.glob("*.jpeg")) + list(ann_dir.glob("*.png"))
        class_counts = defaultdict(int)
        for lbl in labels:
            text = lbl.read_text().strip()
            if not text:
                continue
            for line in text.split("\n"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    name = ALL_PPE_CLASSES.get(cid, f"class_{cid}")
                    class_counts[name] += 1
        return {"annotated_images": len(labels), "total_images": len(images), "class_counts": dict(class_counts)}
    except Exception as e:
        logger.error(f"[Company {company_id}] annotation_status error: {e}")
        raise HTTPException(500, detail=f"Error reading annotations: {str(e)}")


@router.get("/photos/summary", tags=["Photo Upload"], summary="Photo Counts per Category")
async def photo_summary(company_id: int = Depends(get_ui_company)):
    try:
        raw_dir = CompanyData.epi(company_id, "photos_raw")
        summary = {}
        for cls_name in list(ALL_PPE_CLASSES.values()) + ["full_body"]:
            d = raw_dir / cls_name
            n = len(list(d.glob("*.*"))) if d.exists() else 0
            summary[cls_name] = n
        return summary
    except Exception as e:
        logger.error(f"[Company {company_id}] photo_summary error: {e}")
        raise HTTPException(500, detail=f"Error reading photo summary: {str(e)}")


# ======================================================================
# DATASET GENERATION
# ======================================================================
@router.post("/dataset/generate", tags=["Dataset & Training"], summary="Generate YOLOv8 Train/Valid Dataset")
async def generate_dataset(
    train_split: float = Form(0.8),
    company_id: int = Depends(get_ui_company),
):
    try:
        result = epi_engine.generate_dataset(company_id, train_split)
        if "error" in result:
            raise HTTPException(400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] generate_dataset error: {e}")
        raise HTTPException(500, detail=f"Dataset generation failed: {str(e)}")


@router.get("/dataset/stats", tags=["Dataset & Training"], summary="Dataset Image Counts")
async def dataset_stats(company_id: int = Depends(get_ui_company)):
    try:
        train_dir = CompanyData.epi(company_id, "dataset", "train", "images")
        valid_dir = CompanyData.epi(company_id, "dataset", "valid", "images")
        return {
            "train_images": len(list(train_dir.glob("*.*"))) if train_dir.exists() else 0,
            "valid_images": len(list(valid_dir.glob("*.*"))) if valid_dir.exists() else 0,
        }
    except Exception as e:
        logger.error(f"[Company {company_id}] dataset_stats error: {e}")
        raise HTTPException(500, detail=f"Error reading dataset stats: {str(e)}")


# ======================================================================
# TRAINING
# ======================================================================
@router.post("/train/start", tags=["Dataset & Training"], summary="Start Model Training")
async def start_training(req: TrainRequest, company_id: int = Depends(get_ui_company)):
    try:
        return epi_engine.train_model(company_id, req.base_model, req.epochs,
                                       req.batch_size, req.img_size, req.patience)
    except Exception as e:
        logger.error(f"[Company {company_id}] start_training error: {e}")
        raise HTTPException(500, detail=f"Training failed to start: {str(e)}")


@router.get("/train/status", tags=["Dataset & Training"], summary="Poll Training Progress")
async def train_status(company_id: int = Depends(get_ui_company)):
    try:
        return epi_engine.get_train_status(company_id)
    except Exception as e:
        logger.error(f"[Company {company_id}] train_status error: {e}")
        raise HTTPException(500, detail=f"Error getting train status: {str(e)}")


# ======================================================================
# DETECTION — Image
# ======================================================================
@router.post("/detect/image", tags=["Detection — Image"], summary="Detect PPE from Base64 Image (REST API)",
    response_model=APIResponse)
async def detect_image_b64(req: DetectRequest, company_id: int = Depends(get_ui_company)):
    try:
        if not req.image_base64:
            raise HTTPException(400, detail="image_base64 required")
        raw = base64.b64decode(req.image_base64.split(",")[-1])
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, detail="Invalid image — could not decode base64")
        result = epi_engine.detect_image(company_id, img, req.model_name,
                                          req.confidence, req.detect_faces, req.face_threshold)
        if not result["compliant"]:
            await mqtt_client.publish_alert(company_id, "EPI_NON_COMPLIANT", {
                "missing": result["missing"], "camera_id": req.camera_id,
                "faces": result.get("faces", []),
            })
        return APIResponse(data=result, company_id=company_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] detect_image_b64 error: {e}")
        raise HTTPException(500, detail=f"Detection failed: {str(e)}")


@router.post("/detect/upload", tags=["Detection — Image"], summary="Detect PPE from Uploaded Image (UI)")
async def detect_image_upload(
    file: UploadFile = File(...),
    model_name: str = Form("best"),
    confidence: float = Form(0.4),
    detect_faces: bool = Form(False),
    face_threshold: float = Form(0.45),
    company_id: int = Depends(get_ui_company),
):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(400, detail="Empty file uploaded")
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, detail="Invalid image file — could not decode")
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] detect_image_upload error: {e}")
        raise HTTPException(500, detail=f"Detection failed: {str(e)}")


# ======================================================================
# DETECTION — Video
# ======================================================================
@router.post("/detect/video", tags=["Detection — Video"], summary="Process Uploaded Video File")
async def detect_video_upload(
    file: UploadFile = File(...),
    model_name: str = Form("best"),
    confidence: float = Form(0.4),
    skip_frames: int = Form(5),
    detect_faces: bool = Form(False),
    company_id: int = Depends(get_ui_company),
):
    temp_path = None
    try:
        temp_dir = CompanyData.epi(company_id, "temp")
        suffix = Path(file.filename).suffix if file.filename else ".mp4"
        temp_path = temp_dir / f"video_{uuid.uuid4().hex[:8]}{suffix}"
        content = await file.read()
        if not content:
            raise HTTPException(400, detail="Empty video file")
        temp_path.write_bytes(content)
        logger.info(f"[Company {company_id}] Processing video: {file.filename} ({len(content)} bytes)")
        results = epi_engine.process_video(company_id, str(temp_path), model_name,
                                            confidence, skip_frames, detect_faces=detect_faces)
        compliant_count = sum(1 for r in results if r["compliant"])
        total = len(results)
        if total == 0:
            raise HTTPException(400, detail="No frames could be processed. Check if the file is a valid video (MP4, AVI, MOV).")
        return {
            "frames_processed": total,
            "compliant_frames": compliant_count,
            "non_compliant_frames": total - compliant_count,
            "compliance_rate": round(compliant_count / max(total, 1) * 100, 1),
            "details": results[:50],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] detect_video_upload error: {e}")
        raise HTTPException(500, detail=f"Video processing failed: {str(e)}")
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@router.post("/detect/youtube", tags=["Detection — Video"], summary="Process YouTube Video")
async def detect_youtube(
    url: str = Form(...),
    model_name: str = Form("best"),
    confidence: float = Form(0.4),
    max_frames: int = Form(100),
    detect_faces: bool = Form(False),
    company_id: int = Depends(get_ui_company),
):
    temp_path = None
    try:
        import yt_dlp
        temp_dir = CompanyData.epi(company_id, "temp")
        temp_path = temp_dir / f"yt_{uuid.uuid4().hex[:8]}.mp4"
        ydl_opts = {
            "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best",
            "outtmpl": str(temp_path),
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if not temp_path.exists():
            raise HTTPException(400, detail=f"Failed to download YouTube video: {url}")
        results = epi_engine.process_video(company_id, str(temp_path), model_name,
                                            confidence, skip_frames=10, max_frames=max_frames,
                                            detect_faces=detect_faces)
        compliant_count = sum(1 for r in results if r["compliant"])
        total = len(results)
        return {"source": url, "frames_processed": total,
                "compliant_frames": compliant_count,
                "compliance_rate": round(compliant_count / max(total, 1) * 100, 1),
                "details": results[:50]}
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(500, detail="yt-dlp not installed. Run: pip install yt-dlp")
    except Exception as e:
        logger.error(f"[Company {company_id}] detect_youtube error: {e}")
        raise HTTPException(500, detail=f"YouTube processing failed: {str(e)}")
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


# ======================================================================
# FACE RECOGNITION
# ======================================================================
@router.post("/faces/register", tags=["Face Recognition"], summary="Register Person with Face Photo")
async def register_face(
    person_code: str = Form(...),
    person_name: str = Form(...),
    badge_id: str = Form(""),
    file: UploadFile = File(...),
    company_id: int = Depends(get_ui_company),
):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(400, detail="Empty image file")
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, detail="Invalid image — could not decode")
        result = epi_engine.face_engine.register_face(
            company_id, person_code, person_name, badge_id, img,
        )
        return {"success": True, **result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] register_face error: {e}")
        raise HTTPException(500, detail=f"Face registration failed: {str(e)}")


@router.get("/faces/people", tags=["Face Recognition"], summary="List Registered People")
async def list_people(company_id: int = Depends(get_ui_company)):
    try:
        return epi_engine.face_engine.list_people(company_id)
    except Exception as e:
        logger.error(f"[Company {company_id}] list_people error: {e}")
        raise HTTPException(500, detail=f"Error listing people: {str(e)}")


@router.post("/faces/rebuild", tags=["Face Recognition"], summary="Rebuild Face Embeddings Database")
async def rebuild_face_db(company_id: int = Depends(get_ui_company)):
    try:
        db = epi_engine.face_engine.build_face_db(company_id)
        return {"people_loaded": len(db)}
    except Exception as e:
        logger.error(f"[Company {company_id}] rebuild_face_db error: {e}")
        raise HTTPException(500, detail=f"Face DB rebuild failed: {str(e)}")


# ======================================================================
# LIVE STREAMING
# ======================================================================
@router.post("/stream/start", tags=["Live Streaming"], summary="Start Live Stream with Real-Time Detection")
async def start_stream(
    source: str = Form(...),
    source_type: str = Form("rtsp"),
    model_name: str = Form("best"),
    confidence: float = Form(0.4),
    detect_faces: bool = Form(False),
    face_threshold: float = Form(0.45),
    company_id: int = Depends(get_ui_company),
):
    try:
        if not source or not source.strip():
            raise HTTPException(400, detail="Source URL/path is required")
        def process_fn(frame):
            return epi_engine.detect_and_annotate(company_id, frame, model_name, confidence, detect_faces, face_threshold)
        session = stream_manager.create_session(source, source_type, company_id, process_fn)
        session.start()
        return {"session_id": session.session_id, "source": source, "source_type": source_type}
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"[Company {company_id}] start_stream RuntimeError: {e}")
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        logger.error(f"[Company {company_id}] start_stream error: {e}")
        raise HTTPException(500, detail=f"Stream failed to start: {str(e)}")


@router.get("/stream/{session_id}/feed", tags=["Live Streaming"], summary="MJPEG Video Feed")
async def stream_feed(session_id: str):
    session = stream_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, detail=f"Stream session not found: {session_id}")

    def generate():
        while session._running:
            jpeg = stream_manager.get_frame_jpeg(session_id, quality=70)
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/stream/{session_id}/status", tags=["Live Streaming"], summary="Stream Session Status")
async def stream_status(session_id: str):
    try:
        session = stream_manager.get_session(session_id)
        if not session:
            raise HTTPException(404, detail=f"Stream session not found: {session_id}")
        return {**session.info, "latest_result": session.latest_result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"stream_status error for {session_id}: {e}")
        raise HTTPException(500, detail=f"Error getting stream status: {str(e)}")


@router.post("/stream/{session_id}/stop", tags=["Live Streaming"], summary="Stop Stream")
async def stop_stream(session_id: str):
    try:
        stream_manager.stop_session(session_id)
        return {"stopped": True, "session_id": session_id}
    except Exception as e:
        logger.error(f"stop_stream error for {session_id}: {e}")
        raise HTTPException(500, detail=f"Error stopping stream: {str(e)}")


@router.get("/stream/sessions", tags=["Live Streaming"], summary="List Active Stream Sessions")
async def list_streams(company_id: int = Depends(get_ui_company)):
    try:
        return [s for s in stream_manager.list_sessions() if s["company_id"] == company_id]
    except Exception as e:
        logger.error(f"[Company {company_id}] list_streams error: {e}")
        raise HTTPException(500, detail=f"Error listing streams: {str(e)}")


# ======================================================================
# MODELS / RESULTS / STATS
# ======================================================================
@router.get("/models", tags=["Models & Results"], summary="List Trained Models")
async def list_models(company_id: int = Depends(get_ui_company)):
    try:
        return epi_engine.list_models(company_id)
    except Exception as e:
        logger.error(f"[Company {company_id}] list_models error: {e}")
        raise HTTPException(500, detail=f"Error listing models: {str(e)}")


@router.get("/results/{filename}", tags=["Models & Results"], summary="Serve Detection Result Snapshot")
async def get_result_image(filename: str, company_id: int = Depends(get_ui_company)):
    try:
        fpath = CompanyData.epi(company_id, "results", filename)
        if not fpath.exists():
            raise HTTPException(404, detail=f"Result file not found: {filename}")
        return StreamingResponse(open(fpath, "rb"), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] get_result_image error: {e}")
        raise HTTPException(500, detail=f"Error serving result: {str(e)}")


@router.get("/stats", tags=["Models & Results"], summary="System Statistics")
async def epi_stats(company_id: int = Depends(get_ui_company)):
    try:
        config = epi_engine.get_ppe_config(company_id)
        active = epi_engine.get_active_classes(company_id)
        models = epi_engine.list_models(company_id)
        storage = CompanyData.disk_usage(company_id)
        people = epi_engine.face_engine.list_people(company_id)
        return {"company_id": company_id, "ppe_config": config, "active_classes": active,
                "models_count": len(models), "people_count": len(people),
                "storage": storage, "train_status": epi_engine.get_train_status(company_id)}
    except Exception as e:
        logger.error(f"[Company {company_id}] epi_stats error: {e}")
        raise HTTPException(500, detail=f"Error getting stats: {str(e)}")