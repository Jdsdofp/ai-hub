"""
EPI Check API v3 — REST endpoints for PPE detection, face recognition,
annotation converter, training, upload, and streaming.

FIXES v3.1:
  - BUG-05: generate_dataset HTTPException sem `detail=` keyword
  - BUG-06: detect_image_b64 retornava np.int64 não serializável em JSON
  - BUG-07: stream_feed acessava session._running (atributo privado) diretamente

INTEGRAÇÃO MySQL v3.2:
  - detect/upload    → salva em vision_detection_events + vision_epi_detections
  - detect/frame     → salva detecção + alerta se não-conforme
  - detect/image     → salva detecção + alerta se não-conforme
  - detect/video     → salva resumo de detecção + alerta se não-conforme
  - stream/start     → registra sessão em vision_stream_sessions
  - stream/stop      → fecha sessão com frame_count
  - faces/register   → persiste em vision_people + vision_face_photos
  - annotate/save    → persiste em vision_annotations
  - stats            → inclui dados do banco (people_count, alerts_open)
  - GET /analytics/* → 7 endpoints de analytics direto do MySQL

INTEGRAÇÃO MySQL v3.3:
  - detect/upload    → upsert vision_compliance_daily + vision_compliance_hourly
  - detect/frame     → upsert vision_compliance_daily + vision_compliance_hourly
  - detect/video     → upsert vision_compliance_daily + vision_compliance_hourly
  - detect/image     → upsert vision_compliance_daily + vision_compliance_hourly
"""
import base64
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional
from collections import defaultdict
from datetime import date, datetime

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

from app.core.security import get_authenticated_company, get_ui_company
from app.core.company import CompanyData
from app.core.repository import repo
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
# HELPERS
# ======================================================================
def _today() -> str:
    return date.today().isoformat()

def _current_hour() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:00:00")


# ======================================================================
# PPE CONFIGURATION
# ======================================================================
# @router.get("/config", tags=["PPE Configuration"], summary="Get PPE Configuration")
# async def get_ppe_config(company_id: int = Depends(get_ui_company)):
#     try:
#         config = epi_engine.get_ppe_config(company_id)
#         active = epi_engine.get_active_classes(company_id)
#         return {"config": config, "active_classes": active, "all_classes": ALL_PPE_CLASSES}
#     except Exception as e:
#         logger.error(f"[Company {company_id}] get_ppe_config error: {e}")
#         raise HTTPException(500, detail=f"Failed to get PPE config: {str(e)}")
# @router.get("/config", tags=["PPE Configuration"], summary="Get PPE Configuration")
# async def get_ppe_config(company_id: int = Depends(get_ui_company)):
#     try:
#         # Busca do banco primeiro
#         db_config = await repo.get_ppe_config(company_id)
#         if db_config:
#             # Popula cache do engine
#             epi_engine.set_ppe_config_cache(company_id, db_config)
#             config = db_config
#         else:
#             # Banco vazio — usa DEFAULT e salva no banco
#             config = epi_engine.get_ppe_config(company_id)
#             await repo.save_ppe_config(company_id, config)
#             epi_engine.set_ppe_config_cache(company_id, config)

#         active = epi_engine.get_active_classes(company_id)
#         return {"config": config, "active_classes": active, "all_classes": ALL_PPE_CLASSES}
#     except Exception as e:
#         logger.error(f"[Company {company_id}] get_ppe_config error: {e}")
#         raise HTTPException(500, detail=f"Failed to get PPE config: {str(e)}")


@router.get("/config", tags=["PPE Configuration"], summary="Get PPE Configuration")
async def get_ppe_config(company_id: int = Depends(get_ui_company)):
    try:
        # Busca do banco primeiro
        db_config = await repo.get_ppe_config(company_id)

        if db_config:
            config = db_config
        else:
            # Banco vazio — usa DEFAULT e salva no banco
            config = epi_engine.get_ppe_config(company_id)
            await repo.save_ppe_config(company_id, config)

        active = epi_engine.get_active_classes(company_id)

        return {
            "config": config,
            "active_classes": active,
            "all_classes": ALL_PPE_CLASSES
        }

    except Exception as e:
        logger.error(f"[Company {company_id}] get_ppe_config error: {e}")
        raise HTTPException(500, detail=f"Failed to get PPE config: {str(e)}")


@router.post("/config", tags=["PPE Configuration"], summary="Save PPE Configuration")
async def save_ppe_config(config: PPEConfig, company_id: int = Depends(get_ui_company)):
    try:
        # Salva no banco E atualiza cache — disco não é mais usado
        await repo.save_ppe_config(company_id, config.model_dump())
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
                    size_kb = fpath.stat().st_size // 1024
                    results.append({"file": f.filename, "ok": True, "size": f"{w}x{h}"})
                    # ── MySQL ──────────────────────────────────────────────
                    await repo.save_training_photo(
                        company_id=company_id,
                        category=category,
                        filename=f.filename,
                        filepath=str(fpath),
                        width=w, height=h,
                        file_size_kb=size_kb,
                        upload_type="single",
                    )
                    # ───────────────────────────────────────────────────────
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

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", "*.jiff", "*.jfif"}
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
            has_lbl = stem in labels_data
            if has_lbl:
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

            # ── MySQL ──────────────────────────────────────────────────────
            img_path = ann_dir / img_name
            try:
                img_cv = cv2.imread(str(img_path))
                h, w   = img_cv.shape[:2] if img_cv is not None else (None, None)
            except Exception:
                h = w = None
            await repo.save_training_photo(
                company_id=company_id,
                category=category,
                filename=img_name,
                filepath=str(img_path),
                width=w, height=h,
                file_size_kb=len(img_bytes) // 1024,
                has_label=has_lbl,
                upload_type="bulk",
            )
            # ──────────────────────────────────────────────────────────────
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
        active_classes = epi_engine.get_active_classes(company_id)
        lines = []
        for ann in data.annotations:
            cid = int(ann.get('class_id', ann.get('classId', 0)))
            cx = float(ann.get('cx', 0))
            cy = float(ann.get('cy', 0))
            w = float(ann.get('w', 0))
            h = float(ann.get('h', 0))
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        lbl_path.write_text("\n".join(lines))

        # ── MySQL ─────────────────────────────────────────────────────────
        await repo.save_annotations(
            company_id=company_id,
            image_name=data.image_filename,
            annotations=data.annotations,
            active_classes=active_classes,
        )
        # ─────────────────────────────────────────────────────────────────

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
        await repo.save_dataset(
            company_id=company_id,
            train_count=result.get("train_count", 0),
            valid_count=result.get("valid_count", 0),
            classes=result.get("classes", {}),
            yaml_path=result.get("yaml_path", ""),
            train_split=train_split,
        )
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
        result = epi_engine.train_model(company_id, req.base_model, req.epochs,
                                        req.batch_size, req.img_size, req.patience)
        await repo.create_training_run(
            company_id=company_id,
            base_model=req.base_model,
            epochs=req.epochs,
            batch_size=req.batch_size,
            img_size=req.img_size,
            classes=epi_engine.get_active_classes(company_id),
        )
        return result
    except Exception as e:
        logger.error(f"[Company {company_id}] start_training error: {e}")
        raise HTTPException(500, detail=f"Training failed to start: {str(e)}")


@router.get("/train/status", tags=["Dataset & Training"], summary="Poll Training Progress")
async def train_status(company_id: int = Depends(get_ui_company)):
    try:
        status = epi_engine.get_train_status(company_id)
        if status.get("status") in ("complete", "error"):
            rows = await repo.get_training_history(company_id, limit=1)
            if rows and rows[0].get("status") not in ("complete", "error"):
                await repo.update_training_run(
                    run_id=rows[0]["id"],
                    status=status["status"],
                    best_map50=status.get("best_map50"),
                    best_map50_95=status.get("best_map50_95"),
                    model_path=status.get("model_path"),
                    error_message=status.get("error"),
                )
        return status
    except Exception as e:
        logger.error(f"[Company {company_id}] train_status error: {e}")
        raise HTTPException(500, detail=f"Error getting train status: {str(e)}")


# ======================================================================
# UTILS
# ======================================================================
def _sanitize_result(result: dict) -> dict:
    """FIX BUG-06: Converte tipos numpy não-serializáveis para tipos Python nativos."""
    import math

    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if math.isnan(v) or math.isinf(v) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    return _convert(result)


# ======================================================================
# DETECTION — Image (Base64 REST API)
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

        # ── MySQL ─────────────────────────────────────────────────────────
        await repo.save_detection(
            company_id=company_id,
            result=result,
            camera_id=req.camera_id if hasattr(req, "camera_id") else None,
            source_type="upload",
        )
        if not result["compliant"]:
            await mqtt_client.publish_alert(company_id, "EPI_NON_COMPLIANT", {
                "missing": result["missing"], "camera_id": getattr(req, "camera_id", None),
                "faces": result.get("faces", []),
            })
            await repo.save_alert(
                company_id=company_id,
                alert_type="EPI_NON_COMPLIANT",
                details={"missing": result["missing"], "source": "api_b64"},
                severity="high" if len(result.get("missing", [])) > 1 else "medium",
            )
        await repo.upsert_compliance_daily(
            company_id=company_id,
            date=_today(),
            total=1,
            compliant=1 if result["compliant"] else 0,
        )
        await repo.upsert_compliance_hourly(
            company_id=company_id,
            hour_ts=_current_hour(),
            total=1,
            compliant=1 if result["compliant"] else 0,
        )
        # ─────────────────────────────────────────────────────────────────

        return APIResponse(data=_sanitize_result(result), company_id=company_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] detect_image_b64 error: {e}")
        raise HTTPException(500, detail=f"Detection failed: {str(e)}")


# ======================================================================
# DETECTION — Upload (UI)
# ======================================================================
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
        snap_path = snap_dir / snap_name
        cv2.imwrite(str(snap_path), annotated)
        result["snapshot_url"] = f"/api/v1/epi/results/{snap_name}?company_id={company_id}"
        _, buf = cv2.imencode(".jpg", annotated)
        result["annotated_base64"] = base64.b64encode(buf).decode()

        # ── MySQL ─────────────────────────────────────────────────────────
        event_id = await repo.save_detection(
            company_id=company_id,
            result=result,
            snapshot_path=str(snap_path),
            model_name=model_name,
            confidence_threshold=confidence,
            source_type="upload",
        )
        await repo.save_snapshot(
            company_id=company_id,
            filename=snap_name,
            filepath=str(snap_path),
            snapshot_type="EPI_DETECTION",
            source_type="upload",
            event_id=event_id,
        )
        if not result["compliant"]:
            await repo.save_alert(
                company_id=company_id,
                alert_type="EPI_NON_COMPLIANT",
                details={"missing": result["missing"], "snapshot": snap_name},
                severity="high" if len(result.get("missing", [])) > 1 else "medium",
            )
        await repo.upsert_compliance_daily(
            company_id=company_id,
            date=_today(),
            total=1,
            compliant=1 if result["compliant"] else 0,
        )
        await repo.upsert_compliance_hourly(
            company_id=company_id,
            hour_ts=_current_hour(),
            total=1,
            compliant=1 if result["compliant"] else 0,
        )
        # ─────────────────────────────────────────────────────────────────

        return _sanitize_result(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] detect_image_upload error: {e}")
        raise HTTPException(500, detail=f"Detection failed: {str(e)}")


# ======================================================================
# DETECTION — Video
# ======================================================================
# ── Background worker ──────────────────────────────────────────────────────
async def _process_video_job(
    job_id: int,
    company_id: int,
    video_path: str,
    model_name: str,
    confidence: float,
    skip_frames: int,
    detect_faces: bool,
) -> None:
    """
    Roda em background via BackgroundTasks.
    Processa o vídeo frame a frame, atualiza progresso no MySQL,
    e finaliza com complete ou error.
    """
    t0 = time.time()
    temp_path = Path(video_path)
    try:
        # Conta frames para progress tracking
        cap = cv2.VideoCapture(video_path)
        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        expected = max(1, total_frames_raw // max(skip_frames, 1))
        await repo.start_video_job(job_id, frames_total=expected)

        # Processa o vídeo
        results = epi_engine.process_video(
            company_id, video_path, model_name,
            confidence, skip_frames, detect_faces=detect_faces,
        )

        total = len(results)
        if total == 0:
            await repo.fail_video_job(job_id, "No frames could be processed.")
            return

        compliant_count = sum(1 for r in results if r["compliant"])
        non_compliant_frames = [r for r in results if not r["compliant"]]

        # ── Resumo final ──────────────────────────────────────────────
        result_summary = {
            "frames_total": total,
            "frames_compliant": compliant_count,
            "frames_non_compliant": total - compliant_count,
            "compliance_rate": round(compliant_count / total * 100, 1),
            "epi_required_count": results[0].get("required_count", 0) if results else 0,
            "missing_items_sample": non_compliant_frames[0].get("missing", []) if non_compliant_frames else [],
        }

        # ── MySQL ─────────────────────────────────────────────────────
        summary_for_detection = {
            "compliant": compliant_count == total,
            "compliance_score": round(compliant_count / total, 4),
            "epi_required_count": results[0].get("required_count", 0) if results else 0,
            "epi_detected_count": results[0].get("detected_count", 0) if results else 0,
            "epi_missing_count": len(results[0].get("missing", [])) if results else 0,
            "missing_items": non_compliant_frames[0].get("missing", []) if non_compliant_frames else [],
            "detections": non_compliant_frames[0].get("detections", []) if non_compliant_frames else [],
            "faces": [],
        }
        await repo.save_detection(
            company_id=company_id,
            result=summary_for_detection,
            model_name=model_name,
            confidence_threshold=confidence,
            source_type="upload",
        )
        if non_compliant_frames:
            await repo.save_alert(
                company_id=company_id,
                alert_type="EPI_NON_COMPLIANT",
                details={
                    "missing": non_compliant_frames[0].get("missing", []),
                    "source": "video_upload",
                    "job_id": job_id,
                    "frames_non_compliant": len(non_compliant_frames),
                    "total_frames": total,
                    "compliance_rate": result_summary["compliance_rate"],
                },
                severity="high" if len(non_compliant_frames[0].get("missing", [])) > 1 else "medium",
            )
        await repo.upsert_compliance_daily(
            company_id=company_id,
            date=_today(),
            total=total,
            compliant=compliant_count,
        )
        await repo.upsert_compliance_hourly(
            company_id=company_id,
            hour_ts=_current_hour(),
            total=total,
            compliant=compliant_count,
        )

        processing_ms = int((time.time() - t0) * 1000)
        await repo.complete_video_job(
            job_id=job_id,
            frames_total=total,
            frames_processed=total,
            frames_compliant=compliant_count,
            result_summary=result_summary,
            processing_ms=processing_ms,
        )
        logger.info(f"[VideoJob {job_id}] complete — {total} frames, {result_summary['compliance_rate']}% compliant")

    except Exception as e:
        logger.error(f"[VideoJob {job_id}] failed: {e}")
        await repo.fail_video_job(job_id, str(e))
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post(
    "/detect/video",
    tags=["Detection — Video"],
    summary="Submit video for async EPI detection (returns job_id immediately)",
)
async def detect_video_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_name: str = Form("best"),
    confidence: float = Form(0.4),
    skip_frames: int = Form(5),
    detect_faces: bool = Form(False),
    company_id: int = Depends(get_ui_company),
):
    """
    Faz upload do vídeo e retorna job_id imediatamente.
    O processamento ocorre em background.
    Use GET /detect/video/jobs/{job_id} para acompanhar o progresso.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(400, detail="Empty video file")

        # Salva arquivo temporário
        temp_dir = CompanyData.epi(company_id, "temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename).suffix if file.filename else ".mp4"
        temp_path = temp_dir / f"vjob_{uuid.uuid4().hex[:10]}{suffix}"
        temp_path.write_bytes(content)

        # Cria job no MySQL
        job_id = await repo.create_video_job(
            company_id=company_id,
            source_type="file",
            original_name=file.filename,
            model_name=model_name,
            confidence=confidence,
            skip_frames=skip_frames,
            detect_faces=detect_faces,
        )
        if not job_id:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(500, detail="Failed to create video job")

        # Agenda processamento em background
        background_tasks.add_task(
            _process_video_job,
            job_id=job_id,
            company_id=company_id,
            video_path=str(temp_path),
            model_name=model_name,
            confidence=confidence,
            skip_frames=skip_frames,
            detect_faces=detect_faces,
        )

        return {
            "job_id": job_id,
            "status": "pending",
            "original_name": file.filename,
            "model_name": model_name,
            "confidence": confidence,
            "skip_frames": skip_frames,
            "detect_faces": detect_faces,
            "message": f"Processing started. Poll GET /detect/video/jobs/{job_id} for status.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] detect_video_upload error: {e}")
        raise HTTPException(500, detail=f"Video job submission failed: {str(e)}")


@router.get(
    "/detect/video/jobs/{job_id}",
    tags=["Detection — Video"],
    summary="Get status and result of a video processing job",
)
async def get_video_job(
    job_id: int,
    company_id: int = Depends(get_ui_company),
):
    """
    Retorna o estado atual do job.
    - status: pending | processing | complete | error
    - Quando complete: inclui result_summary com compliance_rate e detalhes.
    """
    job = await repo.get_video_job(company_id, job_id)
    if not job:
        raise HTTPException(404, detail=f"Job not found: {job_id}")

    # Calcula progresso percentual
    total = job.get("frames_total") or 0
    processed = job.get("frames_processed") or 0
    progress_pct = round(processed / total * 100, 1) if total else 0

    return _sanitize_result({
        **job,
        "progress_pct": progress_pct,
        "result_summary": json.loads(job["result_summary"])
            if isinstance(job.get("result_summary"), str) else job.get("result_summary"),
    })


@router.get(
    "/detect/video/jobs",
    tags=["Detection — Video"],
    summary="List video processing jobs",
)
async def list_video_jobs(
    status: Optional[str] = Query(None, description="Filter by status: pending | processing | complete | error"),
    limit: int = Query(20, ge=1, le=100),
    company_id: int = Depends(get_ui_company),
):
    """Lista os jobs de vídeo da empresa, ordenados por data de criação (mais recente primeiro)."""
    jobs = await repo.list_video_jobs(company_id, limit=limit, status=status)
    return {
        "total": len(jobs),
        "jobs": _sanitize_result(jobs),
    }


@router.delete(
    "/detect/video/jobs/{job_id}",
    tags=["Detection — Video"],
    summary="Cancel a pending video job",
)
async def cancel_video_job(
    job_id: int,
    company_id: int = Depends(get_ui_company),
):
    """
    Cancela um job com status 'pending'.
    Jobs em 'processing' não podem ser cancelados (já estão rodando em background).
    """
    job = await repo.get_video_job(company_id, job_id)
    if not job:
        raise HTTPException(404, detail=f"Job not found: {job_id}")
    if job.get("status") != "pending":
        raise HTTPException(400, detail=f"Cannot cancel job with status '{job.get('status')}'")
    await repo.fail_video_job(job_id, "Cancelled by user")
    return {"job_id": job_id, "cancelled": True}


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
            raise HTTPException(400, detail=f"Failed to download: {url}")
        results = epi_engine.process_video(company_id, str(temp_path), model_name,
                                            confidence, skip_frames=10, max_frames=max_frames,
                                            detect_faces=detect_faces)
        compliant_count = sum(1 for r in results if r["compliant"])
        total = len(results)
        return {"source": url, "frames_processed": total,
                "compliant_frames": compliant_count,
                "compliance_rate": round(compliant_count / max(total, 1) * 100, 1),
                "details": _sanitize_result(results[:50])}
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(500, detail="yt-dlp not installed.")
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
# DETECTION — Single Frame (Browser Camera)
# ======================================================================
@router.post("/detect/frame", tags=["Detection — Image"],
    summary="Detect PPE from raw JPEG bytes (browser camera frames)")
async def detect_frame(
    file: UploadFile = File(...),
    model_name: str = Form("best"),
    confidence: float = Form(0.4),
    detect_faces: bool = Form(False),
    face_threshold: float = Form(0.45),
    annotate: bool = Form(True),
    company_id: int = Depends(get_ui_company),
):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(400, detail="Empty frame")
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, detail="Invalid frame — could not decode")

        if annotate:
            annotated, result = epi_engine.detect_and_annotate(
                company_id, img, model_name, confidence, detect_faces, face_threshold,
            )
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            result["annotated_base64"] = base64.b64encode(buf).decode()
        else:
            result = epi_engine.detect_image(
                company_id, img, model_name, confidence, detect_faces, face_threshold,
            )

        # ── MySQL ─────────────────────────────────────────────────────────
        await repo.save_detection(
            company_id=company_id,
            result=result,
            model_name=model_name,
            confidence_threshold=confidence,
            source_type="browser_camera",
        )
        if not result["compliant"]:
            await mqtt_client.publish_alert(company_id, "EPI_NON_COMPLIANT", {
                "missing": result["missing"],
                "source": "browser_camera",
                "faces": result.get("faces", []),
            })
            await repo.save_alert(
                company_id=company_id,
                alert_type="EPI_NON_COMPLIANT",
                details={"missing": result["missing"], "source": "browser_camera"},
                severity="high" if len(result.get("missing", [])) > 1 else "medium",
            )
        await repo.upsert_compliance_daily(
            company_id=company_id,
            date=_today(),
            total=1,
            compliant=1 if result["compliant"] else 0,
        )
        await repo.upsert_compliance_hourly(
            company_id=company_id,
            hour_ts=_current_hour(),
            total=1,
            compliant=1 if result["compliant"] else 0,
        )
        # ─────────────────────────────────────────────────────────────────

        return _sanitize_result(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] detect_frame error: {e}")
        raise HTTPException(500, detail=f"Frame detection failed: {str(e)}")


# ======================================================================
# FACE RECOGNITION
# ======================================================================
# @router.post("/faces/register", tags=["Face Recognition"], summary="Register Person with Face Photo")
# async def register_face(
#     person_code: str = Form(...),
#     person_name: str = Form(...),
#     badge_id: str = Form(""),
#     file: UploadFile = File(...),
#     company_id: int = Depends(get_ui_company),
# ):
#     try:
#         data = await file.read()
#         if not data:
#             raise HTTPException(400, detail="Empty image file")
#         arr = np.frombuffer(data, dtype=np.uint8)
#         img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#         if img is None:
#             raise HTTPException(400, detail="Invalid image — could not decode")

#         result = epi_engine.face_engine.register_face(
#             company_id, person_code, person_name, badge_id, img,
#         )

#         # ── MySQL ─────────────────────────────────────────────────────────
#         await repo.upsert_person(
#             company_id=company_id,
#             person_code=person_code,
#             person_name=person_name,
#             badge_id=badge_id,
#         )
#         person_dir = CompanyData.epi(company_id, "people", person_code)
#         photos = sorted(person_dir.glob("face_*.jpg"))
#         if photos:
#             last_photo = photos[-1]
#             await repo.save_face_photo(
#                 company_id=company_id,
#                 person_code=person_code,
#                 filename=last_photo.name,
#                 filepath=str(last_photo),
#             )
#         # ─────────────────────────────────────────────────────────────────

#         return {"success": True, **result}
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"[Company {company_id}] register_face error: {e}")
#         raise HTTPException(500, detail=f"Face registration failed: {str(e)}")

# @router.post("/faces/register", tags=["Face Recognition"], summary="Register Person with Face Photo")
# async def register_face(
#     person_code: str = Form(...),
#     person_name: str = Form(...),
#     badge_id: str = Form(""),
#     file: UploadFile = File(...),
#     company_id: int = Depends(get_ui_company),
# ):
#     try:
#         data = await file.read()
#         if not data:
#             raise HTTPException(400, detail="Empty image file")
#         arr = np.frombuffer(data, dtype=np.uint8)
#         img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#         if img is None:
#             raise HTTPException(400, detail="Invalid image — could not decode")

#         # 1. Gera embedding via face engine (sem salvar no disco)
#         result = epi_engine.face_engine.register_face(
#             company_id, person_code, person_name, badge_id, img,
#         )

#         # 2. Persiste pessoa no MySQL
#         person_id = await repo.upsert_person(
#             company_id=company_id,
#             person_code=person_code,
#             person_name=person_name,
#             badge_id=badge_id,
#         )
#         if person_id is None:
#             raise HTTPException(500, detail="Falha ao salvar pessoa no banco de dados")

#         # 3. Salva foto como base64 no MySQL (sem arquivo em disco)
#         _, buf = cv2.imencode(".jpg", img)
#         img_b64 = base64.b64encode(buf).decode()
#         await repo.save_face_photo(
#             company_id=company_id,
#             person_code=person_code,
#             filename=f"face_{person_code}_{uuid.uuid4().hex[:8]}.jpg",
#             filepath=img_b64,   # armazena b64 no campo filepath
#         )

#         return {"success": True, **result}
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"[Company {company_id}] register_face error: {e}")
#         raise HTTPException(500, detail=f"Face registration failed: {str(e)}")


@router.post("/faces/register", tags=["Face Recognition"], summary="Register Person with Face Photo")
async def register_face(
    person_code: str = Form(...),
    person_name: str = Form(...),
    badge_id: str = Form(""),
    file: UploadFile = File(...),
    company_id: int = Depends(get_ui_company),
):
    """
    Registra uma nova foto de rosto para uma pessoa.
    
    FIX: Armazena arquivo em disco + path em MySQL (não base64)
    Evita: MySQL Error 1406 "Data too long for column 'filepath'"
    
    Args:
        person_code: Código único da pessoa (ex: PERSON_001)
        person_name: Nome da pessoa
        badge_id: ID do crachá (opcional)
        file: Arquivo de imagem (JPG/PNG)
        company_id: ID da empresa
        
    Returns:
        {
            "success": true,
            "person_code": "PERSON_001",
            "photos": 1,
            "quality_score": 0.8542,
            "filepath": "/opt/vision/data/1/epi_check/people/PERSON_001/face_PERSON_001_a1b2c3d4.jpg"
        }
    """
    try:
        # ========== 1. LEITURA E VALIDAÇÃO ==========
        data = await file.read()
        if not data:
            raise HTTPException(400, detail="Empty image file")
        
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, detail="Invalid image — could not decode")
        
        # ========== 2. CÁLCULO DE QUALIDADE (BLUR DETECTION) ==========
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_score = round(min(laplacian_var / 100, 1.0), 4)
        logger.debug(f"[Company {company_id}] Face quality score: {quality_score}")
        
        # ========== 3. GERA EMBEDDING VIA FACE ENGINE ==========
        result = epi_engine.face_engine.register_face(
            company_id, person_code, person_name, badge_id, img,
        )
        
        # ========== 4. PERSISTE PESSOA NO MYSQL ==========
        person_id = await repo.upsert_person(
            company_id=company_id,
            person_code=person_code,
            person_name=person_name,
            badge_id=badge_id,
        )
        if person_id is None:
            raise HTTPException(500, detail="Falha ao salvar pessoa no banco de dados")
        
        # ========== 5. SALVA FOTO EM DISCO (FIX CRÍTICO) ==========
        # Cria diretório da pessoa se não existir
        from pathlib import Path
        person_dir = Path(f"/opt/vision/data/{company_id}/epi_check/people/{person_code}")
        person_dir.mkdir(parents=True, exist_ok=True)
        
        # Define nome e caminho do arquivo
        photo_filename = f"face_{person_code}_{uuid.uuid4().hex[:8]}.jpg"
        photo_path = person_dir / photo_filename
        
        # ⭐ SALVA ARQUIVO REAL EM DISCO (não base64)
        photo_path.write_bytes(data)
        logger.info(f"[Company {company_id}] Face photo saved to disk: {photo_path}")
        
        # ========== 6. PERSISTE PATH EM MYSQL (NÃO BASE64) ==========
        # Agora repo.save_face_photo recebe um caminho (~100 bytes)
        # em vez de base64 gigante (50KB+)
        await repo.save_face_photo(
            company_id=company_id,
            person_code=person_code,
            filename=photo_filename,
            filepath=str(photo_path),  # ⭐ FIX: path string, NÃO base64
        )
        
        # ========== 7. RETORNA RESPOSTA ==========
        response = {
            "success": True,
            "person_code": result.get("person_code"),
            "photos": result.get("photos", 1),
            "quality_score": quality_score,
            "filepath": str(photo_path),
            "message": "Face registered successfully"
        }
        logger.info(f"[Company {company_id}] register_face SUCCESS: {person_code}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] register_face error: {e}", exc_info=True)
        raise HTTPException(500, detail=f"Face registration failed: {str(e)}")


# @router.get("/faces/people", tags=["Face Recognition"], summary="List Registered People")
# async def list_people(company_id: int = Depends(get_ui_company)):
#     try:
#         # MySQL é a fonte primária — reflete edições feitas direto no banco
#         db_people = await repo.list_people(company_id, active_only=False)
#         if db_people:
#             return [
#                 {
#                     "person_code": p["person_code"],
#                     "person_name":  p["person_name"],
#                     "badge_id":     p.get("badge_id") or "",
#                     "photos":       p.get("face_photos_count") or 0,
#                     "active":       bool(p.get("active", True)),
#                     "is_inside":    bool(p.get("is_inside", False)),
#                     "last_entry_at": str(p["last_entry_at"]) if p.get("last_entry_at") else None,
#                 }
#                 for p in db_people
#             ]
#         # Fallback: face engine (disco) caso banco vazio
#         return epi_engine.face_engine.list_people(company_id)
#     except Exception as e:
#         logger.error(f"[Company {company_id}] list_people error: {e}")
#         raise HTTPException(500, detail=f"Error listing people: {str(e)}")

@router.get("/faces/people", tags=["Face Recognition"], summary="List Registered People")
async def list_people(company_id: int = Depends(get_ui_company)):
    try:
        db_people = await repo.list_people(company_id, active_only=False)
        return [
            {
                "person_code":   p["person_code"],
                "person_name":   p["person_name"],
                "badge_id":      p.get("badge_id") or "",
                "photos":        p.get("face_photos_count") or 0,
                "active":        bool(p.get("active", True)),
                "is_inside":     bool(p.get("is_inside", False)),
                "last_entry_at": str(p["last_entry_at"]) if p.get("last_entry_at") else None,
            }
            for p in db_people
        ]
        # fallback de disco REMOVIDO — banco é a única fonte
    except Exception as e:
        logger.error(f"[Company {company_id}] list_people error: {e}")
        print("Error ao salva: ", e)
        raise HTTPException(500, detail=f"Error listing people: {str(e)}")


@router.post("/faces/rebuild", tags=["Face Recognition"], summary="Rebuild Face Embeddings Database")
async def rebuild_face_db(company_id: int = Depends(get_ui_company)):
    try:
        db_result = epi_engine.face_engine.build_face_db(company_id)
        return {"people_loaded": len(db_result)}
    except Exception as e:
        logger.error(f"[Company {company_id}] rebuild_face_db error: {e}")
        raise HTTPException(500, detail=f"Face DB rebuild failed: {str(e)}")


@router.post(
    "/faces/sync",
    tags=["Face Recognition"],
    summary="Sync face engine people to MySQL vision_people",
)
async def sync_faces_to_mysql(company_id: int = Depends(get_ui_company)):
    """
    Le todas as pessoas registradas no face engine (disco) e faz upsert
    em vision_people + vision_face_photos para cada foto encontrada.
    """
    try:
        # Lê direto do people_registry.json — fonte de verdade para pessoas históricas
        registry_path = CompanyData.epi(company_id, "people_registry.json")
        people_registry = {}
        if registry_path.exists():
            import json as _json
            try:
                people_registry = _json.loads(registry_path.read_text())
            except Exception as je:
                logger.warning(f"[faces/sync] Failed to read registry: {je}")

        # Também inclui os do face engine (cadastros recentes)
        engine_people = epi_engine.face_engine.list_people(company_id)
        for ep in engine_people:
            pc = ep.get("person_code") or ep.get("code")
            if pc and pc not in people_registry:
                people_registry[pc] = {
                    "name": ep.get("person_name") or ep.get("name") or "",
                    "badge_id": ep.get("badge_id") or "",
                }

        if not people_registry:
            return {"synced": 0, "skipped": 0, "errors": [], "message": "No people found"}

        synced = 0
        skipped = 0
        errors = []

        for person_code, data in people_registry.items():
            person_name = data.get("name") or data.get("person_name") or ""
            badge_id    = data.get("badge_id") or ""

            if not person_code:
                skipped += 1
                continue

            try:
                await repo.upsert_person(
                    company_id=company_id,
                    person_code=str(person_code),
                    person_name=person_name,
                    badge_id=badge_id,
                )
                person_dir = CompanyData.epi(company_id, "people", str(person_code))
                if person_dir.exists():
                    existing_photos = await repo.get_face_photos(company_id, str(person_code))
                    existing_filenames = {p["filename"] for p in existing_photos}
                    for photo_path in sorted(person_dir.glob("face_*.jpg")):
                        if photo_path.name not in existing_filenames:
                            await repo.save_face_photo(
                                company_id=company_id,
                                person_code=str(person_code),
                                filename=photo_path.name,
                                filepath=str(photo_path),
                            )
                synced += 1
            except Exception as pe:
                errors.append({"person_code": person_code, "error": str(pe)})
                logger.warning(f"[faces/sync] Failed to sync {person_code}: {pe}")

        return {
            "synced": synced,
            "skipped": skipped,
            "errors": errors,
            "total_from_engine": len(people_registry),
            "message": f"{synced} pessoas sincronizadas para o MySQL",
        }
    except Exception as e:
        logger.error(f"[Company {company_id}] sync_faces_to_mysql error: {e}")
        raise HTTPException(500, detail=f"Sync failed: {str(e)}")


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

        # ── MySQL ─────────────────────────────────────────────────────────
        await repo.save_stream_session(
            company_id=company_id,
            session_id=session.session_id,
            source_url=source,
            source_type=source_type,
            model_name=model_name,
            confidence=confidence,
            detect_faces=detect_faces,
        )
        # ─────────────────────────────────────────────────────────────────

        return {"session_id": session.session_id, "source": source, "source_type": source_type}
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        logger.error(f"[Company {company_id}] start_stream error: {e}")
        raise HTTPException(500, detail=f"Stream failed to start: {str(e)}")


@router.get("/stream/{session_id}/feed", tags=["Live Streaming"], summary="MJPEG Video Feed")
async def stream_feed(session_id: str):
    import asyncio
    session = stream_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, detail=f"Stream session not found: {session_id}")

    async def generate():
        while session.info.get("running", False):
            jpeg = stream_manager.get_frame_jpeg(session_id, quality=70)
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/stream/{session_id}/status", tags=["Live Streaming"], summary="Stream Session Status")
async def stream_status(session_id: str):
    try:
        session = stream_manager.get_session(session_id)
        if not session:
            raise HTTPException(404, detail=f"Stream session not found: {session_id}")
        latest = session.latest_result
        return {**session.info, "latest_result": _sanitize_result(latest) if latest else None}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Error getting stream status: {str(e)}")


@router.post("/stream/{session_id}/stop", tags=["Live Streaming"], summary="Stop Stream")
async def stop_stream(session_id: str):
    try:
        session = stream_manager.get_session(session_id)
        frame_count = session.info.get("frame_count", 0) if session else 0
        avg_fps = session.info.get("fps", 0) if session else 0

        stream_manager.stop_session(session_id)

        # ── MySQL ─────────────────────────────────────────────────────────
        await repo.close_stream_session(
            session_id=session_id,
            frame_count=frame_count,
            avg_fps=avg_fps,
        )
        # ─────────────────────────────────────────────────────────────────

        return {"stopped": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(500, detail=f"Error stopping stream: {str(e)}")


@router.get("/stream/sessions", tags=["Live Streaming"], summary="List Active Stream Sessions")
async def list_streams(company_id: int = Depends(get_ui_company)):
    try:
        return [s for s in stream_manager.list_sessions() if s["company_id"] == company_id]
    except Exception as e:
        raise HTTPException(500, detail=f"Error listing streams: {str(e)}")


# ======================================================================
# MODELS / RESULTS / STATS
# ======================================================================
@router.get("/models", tags=["Models & Results"], summary="List Trained Models")
async def list_models(company_id: int = Depends(get_ui_company)):
    try:
        return epi_engine.list_models(company_id)
    except Exception as e:
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
        raise HTTPException(500, detail=f"Error serving result: {str(e)}")


@router.get("/stats", tags=["Models & Results"], summary="System Statistics")
async def epi_stats(company_id: int = Depends(get_ui_company)):
    try:
        config  = epi_engine.get_ppe_config(company_id)
        active  = epi_engine.get_active_classes(company_id)
        models  = epi_engine.list_models(company_id)
        storage = CompanyData.disk_usage(company_id)
        people  = epi_engine.face_engine.list_people(company_id)

        # ── Dados do MySQL ────────────────────────────────────────────────
        db_stats = await repo.get_dashboard_stats(company_id)
        # ─────────────────────────────────────────────────────────────────

        return {
            "company_id":     company_id,
            "ppe_config":     config,
            "active_classes": active,
            "models_count":   len(models),
            "people_count":   db_stats.get("people_count") or len(people),
            "alerts_open":    db_stats.get("alerts_open", 0),
            "storage":        storage,
            "train_status":   epi_engine.get_train_status(company_id),
            "today":          db_stats.get("today", {}),
            "week":           db_stats.get("week", {}),
        }
    except Exception as e:
        logger.error(f"[Company {company_id}] epi_stats error: {e}")
        raise HTTPException(500, detail=f"Error getting stats: {str(e)}")


# ======================================================================
# ANALYTICS — MySQL
# ======================================================================
@router.get("/analytics/dashboard", tags=["Analytics"], summary="Dashboard Stats (MySQL)")
async def analytics_dashboard(company_id: int = Depends(get_ui_company)):
    """Totais hoje, semana, pessoas ativas e alertas abertos."""
    try:
        return await repo.get_dashboard_stats(company_id)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/analytics/compliance/hourly", tags=["Analytics"], summary="Compliance by Hour")
async def analytics_compliance_hourly(
    hours: int = Query(24, ge=1, le=168),
    company_id: int = Depends(get_ui_company),
):
    """Conformidade por hora das últimas N horas (máx 7 dias)."""
    try:
        return await repo.get_hourly_compliance(company_id, hours)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/analytics/missing-ppe", tags=["Analytics"], summary="Missing PPE Ranking")
async def analytics_missing_ppe(
    days: int = Query(7, ge=1, le=90),
    company_id: int = Depends(get_ui_company),
):
    """Ranking dos EPIs mais ausentes nos últimos N dias."""
    try:
        return await repo.get_missing_ppe_ranking(company_id, days)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/analytics/detections", tags=["Analytics"], summary="Recent Detection Events")
async def analytics_detections(
    limit: int = Query(50, ge=1, le=500),
    noncompliant_only: bool = Query(False),
    company_id: int = Depends(get_ui_company),
):
    """Eventos de detecção recentes do banco MySQL."""
    try:
        return await repo.get_recent_detections(company_id, limit, noncompliant_only)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/analytics/alerts", tags=["Analytics"], summary="Alerts (MySQL)")
async def analytics_alerts(
    limit: int = Query(50, ge=1, le=200),
    unresolved_only: bool = Query(False),
    company_id: int = Depends(get_ui_company),
):
    """Alertas registrados no MySQL."""
    try:
        return await repo.get_alerts(company_id, limit, unresolved_only)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/analytics/training-history", tags=["Analytics"], summary="Training History (MySQL)")
async def analytics_training_history(
    limit: int = Query(20, ge=1, le=100),
    company_id: int = Depends(get_ui_company),
):
    """Histórico de treinamentos do MySQL."""
    try:
        return await repo.get_training_history(company_id, limit)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/analytics/people", tags=["Analytics"], summary="People (MySQL)")
async def analytics_people(
    active_only: bool = Query(True),
    company_id: int = Depends(get_ui_company),
):
    """Pessoas registradas no MySQL."""
    try:
        return await repo.list_people(company_id, active_only)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/analytics/compliance/summary", tags=["Analytics"], summary="Compliance Summary")
async def analytics_compliance_summary(
    days: int = Query(7, ge=1, le=90),
    company_id: int = Depends(get_ui_company),
):
    """Resumo de conformidade: total, rate, avg_score dos últimos N dias."""
    try:
        return await repo.get_compliance_summary(company_id, days)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ======================================================================
# TRAIN LOGS
# ======================================================================
@router.get("/train/logs", tags=["Dataset & Training"], summary="Get live training log lines")
async def get_train_logs(
    offset: int = Query(0, ge=0),
    company_id: int = Depends(get_ui_company),
):
    try:
        return epi_engine.get_train_logs(company_id, offset)
    except Exception as e:
        logger.error(f"[Company {company_id}] get_train_logs error: {e}")
        raise HTTPException(500, detail=f"Error reading train logs: {str(e)}")


# ======================================================================
# ZONE ACCESS — visionapp_access_zone_persons
# ======================================================================

@router.get(
    "/access/zones/{zone_id}/persons",
    tags=["Access Control — Zones"],
    summary="List persons with access to a zone",
)
async def list_zone_persons(
    zone_id: int,
    active_only: bool = Query(True),
    company_id: int = Depends(get_ui_company),
):
    """Lista todos os vínculos pessoa-zona para uma zona específica."""
    rows = await repo.list_zone_persons(company_id, zone_id=zone_id, active_only=active_only)
    return _sanitize_result({"zone_id": zone_id, "total": len(rows), "persons": rows})


@router.get(
    "/access/persons/{person_code}/zones",
    tags=["Access Control — Zones"],
    summary="List zones a person has access to",
)
async def list_person_zones(
    person_code: str,
    active_only: bool = Query(True),
    company_id: int = Depends(get_ui_company),
):
    """Lista todas as zonas às quais uma pessoa tem acesso."""
    rows = await repo.list_zone_persons(company_id, person_code=person_code, active_only=active_only)
    return _sanitize_result({"person_code": person_code, "total": len(rows), "zones": rows})


@router.get(
    "/access/zones/{zone_id}/persons/{person_code}/check",
    tags=["Access Control — Zones"],
    summary="Check if a person is allowed in a zone",
)
async def check_zone_access(
    zone_id: int,
    person_code: str,
    company_id: int = Depends(get_ui_company),
):
    """
    Verifica se a pessoa tem permissão na zona.
    Retorna allowed, reason e o perfil de EPI exigido (se houver).
    """
    result = await repo.check_zone_access(company_id, person_code, zone_id)
    return _sanitize_result({"zone_id": zone_id, "person_code": person_code, **result})


@router.post(
    "/access/zones/{zone_id}/persons",
    tags=["Access Control — Zones"],
    summary="Grant or update a person's access to a zone",
)
async def upsert_zone_person(
    zone_id: int,
    person_code: str = Form(...),
    person_name: Optional[str] = Form(None),
    site_id: Optional[int] = Form(None),
    access_allowed: bool = Form(True),
    schedule_id: Optional[int] = Form(None),
    valid_from: Optional[str] = Form(None, description="YYYY-MM-DD"),
    valid_until: Optional[str] = Form(None, description="YYYY-MM-DD"),
    epi_profile_code: Optional[str] = Form(None),
    exposure_limit_min: Optional[float] = Form(None),
    active: bool = Form(True),
    company_id: int = Depends(get_ui_company),
):
    """Cria ou atualiza o vínculo de acesso de uma pessoa a uma zona."""
    ok = await repo.upsert_zone_person(
        company_id=company_id,
        person_code=person_code,
        zone_id=zone_id,
        person_name=person_name,
        site_id=site_id,
        access_allowed=access_allowed,
        schedule_id=schedule_id,
        valid_from=valid_from,
        valid_until=valid_until,
        epi_profile_code=epi_profile_code,
        exposure_limit_min=exposure_limit_min,
        active=active,
    )
    if not ok:
        raise HTTPException(500, detail="Failed to upsert zone person")
    return {"zone_id": zone_id, "person_code": person_code, "saved": True}


@router.patch(
    "/access/zones/{zone_id}/persons/{person_code}/block",
    tags=["Access Control — Zones"],
    summary="Block a person's access to a zone",
)
async def block_zone_person(
    zone_id: int,
    person_code: str,
    reason: str = Form(...),
    company_id: int = Depends(get_ui_company),
):
    """Bloqueia o acesso de uma pessoa a uma zona com motivo registrado."""
    ok = await repo.block_zone_person(company_id, person_code, zone_id, reason)
    if not ok:
        raise HTTPException(500, detail="Failed to block zone person")
    return {"zone_id": zone_id, "person_code": person_code, "blocked": True, "reason": reason}


@router.patch(
    "/access/zones/{zone_id}/persons/{person_code}/unblock",
    tags=["Access Control — Zones"],
    summary="Unblock a person's access to a zone",
)
async def unblock_zone_person(
    zone_id: int,
    person_code: str,
    company_id: int = Depends(get_ui_company),
):
    """Desbloqueia o acesso de uma pessoa a uma zona."""
    ok = await repo.unblock_zone_person(company_id, person_code, zone_id)
    if not ok:
        raise HTTPException(500, detail="Failed to unblock zone person")
    return {"zone_id": zone_id, "person_code": person_code, "blocked": False}


@router.delete(
    "/access/zones/{zone_id}/persons/{person_code}",
    tags=["Access Control — Zones"],
    summary="Remove a person's access to a zone",
)
async def delete_zone_person(
    zone_id: int,
    person_code: str,
    company_id: int = Depends(get_ui_company),
):
    """Remove o vínculo de acesso de uma pessoa a uma zona."""
    ok = await repo.delete_zone_person(company_id, person_code, zone_id)
    if not ok:
        raise HTTPException(500, detail="Failed to delete zone person")
    return {"zone_id": zone_id, "person_code": person_code, "deleted": True}


# ======================================================================
# VALIDATION — Access Control (Face + EPI)
# ======================================================================

def _consolidate_decision(
    photos: list,
    compliance_mode: str,
    photo_count_required: int,
) -> dict:
    """
    Recebe lista de dicts com {face_detected, face_confidence, face_person_code,
    epi_compliant, compliance_score} e retorna:
    {
        access_decision, epi_compliant, compliance_score,
        face_confirmed, face_confidence_max,
        person_code, person_name
    }
    compliance_mode: majority | all | best | worst
    """
    received = len(photos)
    if received == 0:
        return {
            "access_decision": "DENIED_FACE",
            "epi_compliant": False,
            "compliance_score": 0.0,
            "face_confirmed": False,
            "face_confidence_max": None,
            "person_code": None,
            "person_name": None,
        }

    # ── Face consolidation ───────────────────────────────────────────
    face_results = [p for p in photos if p.get("face_detected")]
    face_confirmed = len(face_results) > 0
    face_confidence_max = max((p.get("face_confidence") or 0) for p in photos) or None
    # Melhor reconhecimento
    recognized = [p for p in face_results if p.get("face_person_code")]
    person_code = recognized[0].get("face_person_code") if recognized else None
    person_name = recognized[0].get("face_person_name") if recognized else None

    # ── EPI consolidation ────────────────────────────────────────────
    epi_scores = [p.get("compliance_score") or 0.0 for p in photos]
    epi_flags  = [bool(p.get("epi_compliant")) for p in photos]

    if compliance_mode == "all":
        epi_compliant = all(epi_flags)
    elif compliance_mode == "best":
        epi_compliant = any(epi_flags)
    elif compliance_mode == "worst":
        epi_compliant = all(epi_flags)
    else:  # majority (default)
        epi_compliant = sum(epi_flags) > (received / 2)

    compliance_score = round(sum(epi_scores) / received, 4) if epi_scores else 0.0

    # ── Access decision ──────────────────────────────────────────────
    if not face_confirmed:
        access_decision = "DENIED_FACE"
    elif not epi_compliant:
        access_decision = "DENIED_EPI"
    else:
        access_decision = "GRANTED"

    return {
        "access_decision": access_decision,
        "epi_compliant": epi_compliant,
        "compliance_score": compliance_score,
        "face_confirmed": face_confirmed,
        "face_confidence_max": face_confidence_max,
        "person_code": person_code,
        "person_name": person_name,
    }


@router.post(
    "/validation/start",
    tags=["Validation"],
    summary="Start a new validation session for a door",
)
async def validation_start(
    door_id: str = Form(...),
    direction: str = Form("ENTRY"),              # ENTRY | EXIT
    compliance_mode: str = Form("majority"),     # majority | all | best | worst
    photo_count_required: int = Form(3),
    timeout_seconds: int = Form(30),
    camera_id: Optional[int] = Form(None),
    zone_id: Optional[int] = Form(None),
    company_id: int = Depends(get_ui_company),
):
    """
    Abre uma sessão de validação.
    Retorna session_uuid para ser usado nas chamadas subsequentes.
    Expira automaticamente após timeout_seconds.
    """
    try:
        # Expira sessões antigas antes de abrir nova
        await repo.expire_timed_out_sessions(company_id)

        session_uuid = str(uuid.uuid4())
        session_id = await repo.create_validation_session(
            company_id=company_id,
            session_uuid=session_uuid,
            door_id=door_id,
            direction=direction,
            timeout_seconds=timeout_seconds,
            camera_id=camera_id,
            zone_id=zone_id,
            compliance_mode=compliance_mode,
            photo_count_required=photo_count_required,
        )
        if not session_id:
            raise HTTPException(500, detail="Failed to create validation session")

        return {
            "session_uuid": session_uuid,
            "session_id": session_id,
            "door_id": door_id,
            "direction": direction,
            "compliance_mode": compliance_mode,
            "photo_count_required": photo_count_required,
            "timeout_seconds": timeout_seconds,
            "status": "open",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] validation_start error: {e}")
        raise HTTPException(500, detail=f"Validation start failed: {str(e)}")


@router.post(
    "/validation/photo",
    tags=["Validation"],
    summary="Submit a photo for an open validation session",
)
async def validation_photo(
    session_uuid: str = Form(...),
    file: UploadFile = File(...),
    model_name: str = Form("best"),
    confidence: float = Form(0.4),
    face_threshold: float = Form(0.45),
    company_id: int = Depends(get_ui_company),
):
    """
    Processa uma foto dentro de uma sessão de validação.
    Executa face recognition + EPI detection na mesma imagem.
    Salva resultado em vision_validation_photos.
    Retorna estado atual da sessão e se já completou o número de fotos requerido.
    """
    try:
        # 1. Busca sessão
        session = await repo.get_validation_session(company_id, session_uuid)
        if not session:
            raise HTTPException(404, detail=f"Session not found: {session_uuid}")
        if session.get("session_status") != "open":
            raise HTTPException(400, detail=f"Session is not open: {session.get('session_status')}")

        # Verifica timeout
        from datetime import datetime as _dt
        expires_at = session.get("expires_at")
        if expires_at and _dt.utcnow() > expires_at:
            await repo.close_validation_session(
                session_id=session["id"],
                company_id=company_id,
                session_status="timeout",
                access_decision="DENIED_FACE",
            )
            raise HTTPException(408, detail="Session expired")

        session_id = session["id"]
        photo_count_required = session.get("photo_count_required", 3)

        # 2. Decodifica imagem
        data = await file.read()
        if not data:
            raise HTTPException(400, detail="Empty frame")
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, detail="Invalid image — could not decode")

        h, w = img.shape[:2]
        file_size_kb = len(data) // 1024
        t0 = time.time()

        # 3. EPI + Face detection
        annotated, epi_result = epi_engine.detect_and_annotate(
            company_id, img, model_name, confidence,
            detect_faces=True, face_threshold=face_threshold,
        )
        processing_ms = int((time.time() - t0) * 1000)

        # 4. Extrai dados de face
        faces = epi_result.get("faces", [])
        face_detected = len(faces) > 0
        best_face = max(faces, key=lambda f: f.get("confidence", 0)) if faces else {}
        face_confidence = best_face.get("confidence")
        face_person_code = best_face.get("person_code") if best_face.get("recognized") else None
        face_bbox = best_face.get("bbox")  # espera dict {x,y,w,h} ou None

        # 5. Extrai dados de EPI
        epi_compliant = bool(epi_result.get("compliant", False))
        compliance_score = float(epi_result.get("compliance_score") or 0.0)
        missing = epi_result.get("missing", [])
        detections = epi_result.get("detections", [])
        epi_required = epi_result.get("epi_required_count", 0)
        epi_detected = epi_result.get("epi_detected_count", 0)
        epi_missing = len(missing)

        # 6. Salva snapshot da foto anotada
        snap_dir = CompanyData.epi(company_id, "validation")
        snap_dir.mkdir(parents=True, exist_ok=True)
        photo_seq_current = (session.get("photo_count_received") or 0) + 1
        snap_name = f"val_{session_uuid[:8]}_p{photo_seq_current}_{uuid.uuid4().hex[:6]}.jpg"
        snap_path = snap_dir / snap_name
        cv2.imwrite(str(snap_path), annotated)

        # 7. Persiste foto no MySQL
        await repo.add_validation_photo(
            company_id=company_id,
            session_id=session_id,
            session_uuid=session_uuid,
            photo_seq=photo_seq_current,
            filename=snap_name,
            filepath=str(snap_path),
            file_size_kb=file_size_kb,
            width=w, height=h,
            face_detected=face_detected,
            face_confidence=face_confidence,
            face_person_code=face_person_code,
            face_bbox=face_bbox if isinstance(face_bbox, dict) else None,
            epi_compliant=epi_compliant,
            compliance_score=compliance_score,
            epi_units_required=epi_required,
            epi_units_detected=epi_detected,
            epi_units_missing=epi_missing,
            compliance_detail={"missing": missing},
            raw_detections=detections,
            model_name=model_name,
            processing_ms=processing_ms,
        )

        # 8. Incrementa contador
        new_count = await repo.increment_photo_count(session_id, company_id)

        # 9. Monta anotação base64 para retorno
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        annotated_b64 = base64.b64encode(buf).decode()

        # 10. Se atingiu o número requerido → consolida e fecha sessão
        session_complete = new_count >= photo_count_required
        final_decision = None
        if session_complete:
            photos_rows = await repo.get_validation_photos(company_id, session_uuid)
            # Monta lista para consolidação
            photos_list = [
                {
                    "face_detected":    r.get("face_detected", False),
                    "face_confidence":  r.get("face_confidence"),
                    "face_person_code": r.get("face_person_code"),
                    "epi_compliant":    r.get("epi_compliant", False),
                    "compliance_score": r.get("compliance_score", 0.0),
                }
                for r in photos_rows
            ]
            decision = _consolidate_decision(
                photos_list,
                compliance_mode=session.get("compliance_mode", "majority"),
                photo_count_required=photo_count_required,
            )

            # ── Zone access check ─────────────────────────────────────
            # Se face foi reconhecida e a sessão tem zone_id configurado,
            # valida permissão na tabela visionapp_access_zone_persons.
            zone_id_sess = session.get("zone_id")
            recognized_person = decision.get("person_code")
            if (
                decision["access_decision"] == "GRANTED"
                and zone_id_sess
                and recognized_person
            ):
                zone_check = await repo.check_zone_access(
                    company_id, recognized_person, zone_id_sess
                )
                if not zone_check["allowed"]:
                    reason_map = {
                        "BLOCKED":       "DENIED_FACE",
                        "EXPIRED":       "DENIED_SCHEDULE",
                        "NO_PERMISSION": "DENIED_SCHEDULE",
                        "INACTIVE":      "DENIED_SCHEDULE",
                    }
                    decision["access_decision"] = reason_map.get(
                        zone_check["reason"], "DENIED_SCHEDULE"
                    )
                    decision["zone_denied_reason"] = zone_check["reason"]
                    decision["zone_blocked_reason"] = zone_check.get("blocked_reason")

            final_decision = decision
            await repo.close_validation_session(
                session_id=session_id,
                company_id=company_id,
                session_status="complete",
                access_decision=decision["access_decision"],
                epi_compliant=decision.get("epi_compliant"),
                compliance_score=decision.get("compliance_score"),
                face_confirmed=decision.get("face_confirmed"),
                face_confidence_max=decision.get("face_confidence_max"),
                person_code=decision.get("person_code"),
                person_name=decision.get("person_name"),
            )
            # Publica alerta MQTT se negado
            if decision["access_decision"] != "GRANTED":
                await mqtt_client.publish_alert(company_id, "ACCESS_DENIED", {
                    "door_id": session.get("door_id"),
                    "zone_id": zone_id_sess,
                    "decision": decision["access_decision"],
                    "session_uuid": session_uuid,
                })

        return _sanitize_result({
            "session_uuid": session_uuid,
            "photo_seq": photo_seq_current,
            "photo_count_received": new_count,
            "photo_count_required": photo_count_required,
            "session_complete": session_complete,
            "face_detected": face_detected,
            "face_confidence": face_confidence,
            "face_person_code": face_person_code,
            "epi_compliant": epi_compliant,
            "compliance_score": compliance_score,
            "missing": missing,
            "processing_ms": processing_ms,
            "annotated_base64": annotated_b64,
            "final_decision": final_decision,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] validation_photo error: {e}")
        raise HTTPException(500, detail=f"Validation photo failed: {str(e)}")


@router.post(
    "/validation/close",
    tags=["Validation"],
    summary="Force-close a validation session before photo quota",
)
async def validation_close(
    session_uuid: str = Form(...),
    company_id: int = Depends(get_ui_company),
):
    """
    Fecha a sessão manualmente com as fotos recebidas até agora.
    Útil quando o hardware precisa liberar a câmera antes de completar as 3 fotos.
    """
    try:
        session = await repo.get_validation_session(company_id, session_uuid)
        if not session:
            raise HTTPException(404, detail=f"Session not found: {session_uuid}")
        if session.get("session_status") != "open":
            return {"session_uuid": session_uuid, "status": session.get("session_status")}

        photos_rows = await repo.get_validation_photos(company_id, session_uuid)
        photos_list = [
            {
                "face_detected":    r.get("face_detected", False),
                "face_confidence":  r.get("face_confidence"),
                "face_person_code": r.get("face_person_code"),
                "epi_compliant":    r.get("epi_compliant", False),
                "compliance_score": r.get("compliance_score", 0.0),
            }
            for r in photos_rows
        ]
        decision = _consolidate_decision(
            photos_list,
            compliance_mode=session.get("compliance_mode", "majority"),
            photo_count_required=session.get("photo_count_required", 3),
        )
        await repo.close_validation_session(
            session_id=session["id"],
            company_id=company_id,
            session_status="complete",
            **decision,
        )
        return {
            "session_uuid": session_uuid,
            "status": "complete",
            "photos_used": len(photos_list),
            **decision,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] validation_close error: {e}")
        raise HTTPException(500, detail=f"Validation close failed: {str(e)}")


@router.get(
    "/validation/{session_uuid}/status",
    tags=["Validation"],
    summary="Get current status of a validation session",
)
async def validation_status(
    session_uuid: str,
    company_id: int = Depends(get_ui_company),
):
    """Retorna o estado atual da sessão, incluindo fotos já processadas."""
    try:
        session = await repo.get_validation_session(company_id, session_uuid)
        if not session:
            raise HTTPException(404, detail=f"Session not found: {session_uuid}")
        photos = await repo.get_validation_photos(company_id, session_uuid)
        return _sanitize_result({
            "session_uuid": session_uuid,
            "door_id": session.get("door_id"),
            "direction": session.get("direction"),
            "session_status": session.get("session_status"),
            "access_decision": session.get("access_decision"),
            "photo_count_required": session.get("photo_count_required"),
            "photo_count_received": session.get("photo_count_received"),
            "epi_compliant": session.get("epi_compliant"),
            "compliance_score": session.get("compliance_score"),
            "face_confirmed": session.get("face_confirmed"),
            "person_code": session.get("person_code"),
            "person_name": session.get("person_name"),
            "expires_at": str(session.get("expires_at")) if session.get("expires_at") else None,
            "closed_at": str(session.get("closed_at")) if session.get("closed_at") else None,
            "photos": [
                {
                    "photo_seq": p.get("photo_seq"),
                    "face_detected": p.get("face_detected"),
                    "face_confidence": p.get("face_confidence"),
                    "face_person_code": p.get("face_person_code"),
                    "epi_compliant": p.get("epi_compliant"),
                    "compliance_score": p.get("compliance_score"),
                    "processing_ms": p.get("processing_ms"),
                }
                for p in photos
            ],
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] validation_status error: {e}")
        raise HTTPException(500, detail=f"Validation status failed: {str(e)}")


@router.delete(
    "/validation/{session_uuid}",
    tags=["Validation"],
    summary="Cancel / abort a validation session",
)
async def validation_cancel(
    session_uuid: str,
    company_id: int = Depends(get_ui_company),
):
    """Cancela uma sessão aberta sem processar resultado."""
    try:
        session = await repo.get_validation_session(company_id, session_uuid)
        if not session:
            raise HTTPException(404, detail=f"Session not found: {session_uuid}")
        await repo.close_validation_session(
            session_id=session["id"],
            company_id=company_id,
            session_status="error",
            access_decision="DENIED_FACE",
        )
        return {"session_uuid": session_uuid, "cancelled": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Company {company_id}] validation_cancel error: {e}")
        raise HTTPException(500, detail=f"Validation cancel failed: {str(e)}")