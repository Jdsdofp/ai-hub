"""
EPI Detection Engine v3 — YOLOv8 + Face Recognition + Converter.
Per-company models, configurable PPE classes, polygon converter, face DB.
"""
import time
import json
import threading
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from app.core.config import settings
from app.core.company import CompanyData

# Master PPE class registry
ALL_PPE_CLASSES = {
    0: "thermal_coat", 1: "thermal_pants", 2: "gloves",
    3: "helmet", 4: "boots", 5: "person",
}

ALL_CLASS_COLORS = {
    "thermal_coat": (0, 0, 255), "thermal_pants": (255, 0, 0),
    "gloves": (0, 255, 0), "helmet": (0, 255, 255),
    "boots": (255, 0, 255), "person": (255, 165, 0),
}

DEFAULT_PPE_CONFIG = {
    "thermal_coat": False, "thermal_pants": False, "gloves": False,
    "helmet": True, "boots": True, "person": True,
}


class FaceEngine:
    """InsightFace-based face recognition per company."""

    def __init__(self):
        self._app = None
        self._face_dbs: Dict[int, dict] = {}  # company_id -> {name: embedding}
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._app is not None:
            return
        with self._lock:
            if self._app is not None:
                return
            try:
                import insightface
                self._app = insightface.app.FaceAnalysis(
                    name=settings.FACE_MODEL,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                self._app.prepare(ctx_id=0, det_size=(settings.FACE_DET_SIZE, settings.FACE_DET_SIZE))
                logger.info("InsightFace model loaded")
            except Exception as e:
                logger.error(f"InsightFace load failed: {e}")
                self._app = None

    def build_face_db(self, company_id: int) -> dict:
        """Build face database from registered people photos."""
        self._ensure_loaded()
        if self._app is None:
            return {}

        people_dir = CompanyData.epi(company_id, "people")
        face_db = {}

        if not people_dir.exists():
            return face_db

        for person_folder in sorted(people_dir.iterdir()):
            if not person_folder.is_dir():
                continue

            embeddings = []
            for img_file in person_folder.glob("*.*"):
                if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                faces = self._app.get(img)
                if faces:
                    embeddings.append(faces[0].embedding)

            if embeddings:
                face_db[person_folder.name] = np.mean(embeddings, axis=0)

            # Load registry for display name
            registry = self._load_registry(company_id)
            reg_info = registry.get(person_folder.name, {})
            if reg_info:
                face_db[person_folder.name] = {
                    "embedding": np.mean(embeddings, axis=0) if embeddings else None,
                    "name": reg_info.get("name", person_folder.name),
                    "badge_id": reg_info.get("badge_id", ""),
                }
            elif embeddings:
                face_db[person_folder.name] = {
                    "embedding": np.mean(embeddings, axis=0),
                    "name": person_folder.name,
                    "badge_id": "",
                }

        self._face_dbs[company_id] = face_db
        logger.info(f"[Company {company_id}] Face DB: {len(face_db)} people")
        return face_db

    def recognize_faces(self, company_id: int, img: np.ndarray, face_threshold: float = 0.0) -> list:
        """Detect and recognize faces in an image."""
        self._ensure_loaded()
        if self._app is None:
            return []

        threshold = face_threshold if face_threshold > 0 else settings.FACE_SIMILARITY_THRESHOLD

        if company_id not in self._face_dbs:
            self.build_face_db(company_id)

        face_db = self._face_dbs.get(company_id, {})
        faces_detected = self._app.get(img)
        results = []

        for face in faces_detected:
            emb = face.embedding
            best_name = "UNKNOWN"
            best_display = "UNKNOWN"
            best_score = 0.0
            best_badge = ""

            for db_key, db_info in face_db.items():
                if isinstance(db_info, dict):
                    db_emb = db_info.get("embedding")
                    display = db_info.get("name", db_key)
                    badge = db_info.get("badge_id", "")
                else:
                    db_emb = db_info
                    display = db_key
                    badge = ""

                if db_emb is None:
                    continue

                from numpy.linalg import norm
                score = float(np.dot(emb, db_emb) / (norm(emb) * norm(db_emb)))
                if score > best_score and score > threshold:
                    best_score = score
                    best_name = db_key
                    best_display = display
                    best_badge = badge

            x1, y1, x2, y2 = face.bbox.astype(int)
            results.append({
                "recognized": best_name != "UNKNOWN",
                "person_code": best_name,
                "person_name": best_display,
                "badge_id": best_badge,
                "confidence": round(best_score, 4),
                "bbox": {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)},
            })

        return results

    def register_face(self, company_id: int, person_code: str, person_name: str,
                       badge_id: str, img: np.ndarray) -> dict:
        """Register a new face photo for a person."""
        person_dir = CompanyData.epi(company_id, "people", person_code)
        person_dir.mkdir(parents=True, exist_ok=True)

        n = len(list(person_dir.glob("*.*")))
        img_path = person_dir / f"face_{n + 1:03d}.jpg"
        cv2.imwrite(str(img_path), img)

        # Update registry
        registry = self._load_registry(company_id)
        registry[person_code] = {
            "name": person_name,
            "badge_id": badge_id,
            "folder": str(person_dir),
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save_registry(company_id, registry)

        # Rebuild face DB
        self.build_face_db(company_id)

        return {"person_code": person_code, "photos": n + 1}

    def list_people(self, company_id: int) -> list:
        registry = self._load_registry(company_id)
        result = []
        for code, info in registry.items():
            folder = CompanyData.epi(company_id, "people", code)
            n_photos = len(list(folder.glob("*.*"))) if folder.exists() else 0
            result.append({
                "person_code": code,
                "person_name": info.get("name", code),
                "badge_id": info.get("badge_id", ""),
                "photos": n_photos,
            })
        return result

    def _load_registry(self, company_id: int) -> dict:
        path = CompanyData.epi(company_id, "people_registry.json")
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def _save_registry(self, company_id: int, registry: dict):
        path = CompanyData.epi(company_id, "people_registry.json")
        path.write_text(json.dumps(registry, indent=2))


class AnnotationConverter:
    """Converts polygon/OBB annotations to YOLO bbox format."""

    @staticmethod
    def convert_line(line: str, remap: dict = None) -> Optional[str]:
        parts = line.strip().split()
        if len(parts) < 5:
            return None
        cls_id = int(parts[0])
        if remap:
            cls_id = remap.get(cls_id, cls_id)
        coords = [float(x) for x in parts[1:]]

        if len(parts) == 5:
            return f"{cls_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}"
        elif len(parts) >= 9 and len(parts) % 2 == 1:
            xs = [coords[i] for i in range(0, len(coords), 2)]
            ys = [coords[i] for i in range(1, len(coords), 2)]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        return None

    @staticmethod
    def convert_file(input_path: str, output_path: str, remap: dict = None) -> dict:
        with open(input_path) as f:
            lines = f.readlines()
        converted = []
        for line in lines:
            if not line.strip():
                continue
            result = AnnotationConverter.convert_line(line, remap)
            if result:
                converted.append(result)
        with open(output_path, "w") as f:
            f.write("\n".join(converted))
        return {"converted": len(converted)}

    @staticmethod
    def convert_directory(input_dir: str, output_dir: str, remap: dict = None,
                          copy_images: bool = True) -> dict:
        import glob
        import shutil
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        txt_files = sorted(glob.glob(input_dir + "/*.txt"))
        img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        total_files = 0
        total_boxes = 0

        for txt_path in txt_files:
            stem = Path(txt_path).stem
            out_txt = str(Path(output_dir) / (stem + ".txt"))
            result = AnnotationConverter.convert_file(txt_path, out_txt, remap)
            total_files += 1
            total_boxes += result["converted"]

            if copy_images:
                for ext in img_exts:
                    img_path = str(Path(input_dir) / (stem + ext))
                    if Path(img_path).exists():
                        shutil.copy2(img_path, output_dir)
                        break

        return {"files": total_files, "boxes": total_boxes}


class EPIEngine:
    """Per-company EPI detection using YOLOv8 with face recognition."""

    def __init__(self):
        self._models: Dict[str, object] = {}
        self._lock = threading.Lock()
        self._train_status: Dict[int, dict] = {}
        self.face_engine = FaceEngine()
        self.converter = AnnotationConverter()

    # --- PPE Config ---
    def get_ppe_config(self, company_id: int) -> dict:
        cfg_path = CompanyData.epi(company_id, "ppe_config.json")
        if cfg_path.exists():
            return json.loads(cfg_path.read_text())
        return dict(DEFAULT_PPE_CONFIG)

    def save_ppe_config(self, company_id: int, config: dict):
        cfg_path = CompanyData.epi(company_id, "ppe_config.json")
        cfg_path.write_text(json.dumps(config, indent=2))

    def get_active_classes(self, company_id: int) -> dict:
        config = self.get_ppe_config(company_id)
        return {cid: name for cid, name in ALL_PPE_CLASSES.items() if config.get(name, False)}

    def get_remapped_classes(self, company_id: int) -> Tuple[dict, dict]:
        active = self.get_active_classes(company_id)
        remapped = {}
        remap_table = {}
        for new_id, (old_id, name) in enumerate(sorted(active.items())):
            remapped[new_id] = name
            remap_table[old_id] = new_id
        return remapped, remap_table

    # --- Model management ---
    def _model_key(self, company_id: int, model_name: str) -> str:
        return f"{company_id}:{model_name}"

    def _find_model_path(self, company_id: int, model_name: str) -> Optional[Path]:
        root = CompanyData.epi(company_id, "models")
        for p in [root / f"{model_name}.pt",
                   root / model_name / "weights" / "best.pt",
                   root / "epi_detector" / "weights" / "best.pt"]:
            if p.exists():
                return p
        shared = Path(settings.DATA_ROOT) / "shared" / "models" / f"{model_name}.pt"
        return shared if shared.exists() else None

    def _load_model(self, company_id: int, model_name: str = "best"):
        key = self._model_key(company_id, model_name)
        if key in self._models:
            return self._models[key]
        with self._lock:
            if key in self._models:
                return self._models[key]
            from ultralytics import YOLO
            path = self._find_model_path(company_id, model_name)
            if path:
                model = YOLO(str(path))
                logger.info(f"[Company {company_id}] Loaded model: {path}")
            else:
                model = YOLO("yolov8n.pt")
                logger.warning(f"[Company {company_id}] Model '{model_name}' not found, using yolov8n")
            self._models[key] = model
            return model

    # --- Detection ---
    def detect_image(self, company_id: int, img: np.ndarray,
                     model_name: str = "best", confidence: float = 0.4,
                     detect_faces: bool = False, face_threshold: float = 0.0) -> dict:
        t0 = time.time()
        model = self._load_model(company_id, model_name)
        config = self.get_ppe_config(company_id)

        results = model.predict(
            source=img, conf=confidence,
            imgsz=settings.EPI_INPUT_SIZE, verbose=False,
            device=settings.GPU_DEVICE if settings.GPU_ENABLED else "cpu",
        )

        detections = []
        detected_names = set()
        model_names = model.names or {}

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model_names.get(cls_id, f"class_{cls_id}").lower()
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    "class_name": cls_name, "confidence": round(conf, 4),
                    "bbox": {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)},
                })
                detected_names.add(cls_name)

        required = {n for n in config if config.get(n) and n != "person"}
        missing = sorted(required - detected_names)

        # Face recognition
        faces = []
        if detect_faces:
            faces = self.face_engine.recognize_faces(company_id, img, face_threshold)

        return {
            "compliant": len(missing) == 0,
            "required_count": len(required),
            "detected_count": len(required) - len(missing),
            "missing": missing,
            "detections": detections,
            "faces": faces,
            "model_name": model_name,
            "processing_ms": int((time.time() - t0) * 1000),
        }

    def detect_and_annotate(self, company_id: int, img: np.ndarray,
                             model_name: str = "best", confidence: float = 0.4,
                             detect_faces: bool = False, face_threshold: float = 0.0) -> Tuple[np.ndarray, dict]:
        result = self.detect_image(company_id, img, model_name, confidence, detect_faces, face_threshold)
        annotated = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for det in result["detections"]:
            b = det["bbox"]
            color = ALL_CLASS_COLORS.get(det["class_name"], (255, 255, 0))
            cv2.rectangle(annotated, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), color, 3)
            label = f"{det['class_name']} {det['confidence']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
            cv2.rectangle(annotated, (b["x"], max(b["y"] - th - 8, 0)),
                          (b["x"] + tw + 6, b["y"]), color, -1)
            cv2.putText(annotated, label, (b["x"] + 3, max(b["y"] - 4, th + 4)),
                        font, 0.6, (255, 255, 255), 2)

        # Draw face boxes
        for face in result.get("faces", []):
            fb = face["bbox"]
            fc = (0, 255, 0) if face["recognized"] else (0, 0, 255)
            cv2.rectangle(annotated, (fb["x"], fb["y"]),
                          (fb["x"] + fb["w"], fb["y"] + fb["h"]), fc, 2)
            fl = f"{face['person_name']} ({face['confidence']:.0%})"
            cv2.putText(annotated, fl, (fb["x"], fb["y"] - 10), font, 0.7, fc, 2)

        # Status banner
        person_label = ""
        if result.get("faces"):
            best_face = max(result["faces"], key=lambda f: f["confidence"])
            person_label = f" — {best_face['person_name']}"

        if result["compliant"]:
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 45), (0, 180, 0), -1)
            cv2.putText(annotated, f"COMPLIANT{person_label}", (10, 30), font, 0.8, (255, 255, 255), 2)
        else:
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 45), (0, 0, 255), -1)
            txt = f"NON-COMPLIANT{person_label} — Missing: {', '.join(result['missing'])}"
            cv2.putText(annotated, txt, (10, 30), font, 0.7, (255, 255, 255), 2)

        return annotated, result

    # --- Video ---
    def process_video(self, company_id: int, video_path: str, model_name: str = "best",
                      confidence: float = 0.4, skip_frames: int = 5,
                      max_frames: int = 0, detect_faces: bool = False) -> list:
        cap = cv2.VideoCapture(video_path)
        results_list = []
        frame_num = 0
        processed = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            if frame_num % (skip_frames + 1) != 0:
                continue
            result = self.detect_image(company_id, frame, model_name, confidence, detect_faces)
            result["frame_number"] = frame_num
            results_list.append(result)
            processed += 1
            if 0 < max_frames <= processed:
                break
        cap.release()
        return results_list

    # --- Training ---
    def get_train_status(self, company_id: int) -> dict:
        return self._train_status.get(company_id, {"status": "idle"})

    def train_model(self, company_id: int, base_model: str = "yolov8m.pt",
                    epochs: int = 60, batch_size: int = 16,
                    img_size: int = 640, patience: int = 15) -> dict:
        if self._train_status.get(company_id, {}).get("status") == "training":
            return {"status": "error", "error": "Training already in progress"}
        t = threading.Thread(target=self._train_worker,
                             args=(company_id, base_model, epochs, batch_size, img_size, patience),
                             daemon=True)
        self._train_status[company_id] = {"status": "preparing",
                                           "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        t.start()
        return self._train_status[company_id]

    def _train_worker(self, company_id, base_model, epochs, batch_size, img_size, patience):
        try:
            from ultralytics import YOLO
            yaml_path = str(CompanyData.epi(company_id, "dataset", "data.yaml").resolve())
            if not Path(yaml_path).exists():
                self._train_status[company_id] = {"status": "error",
                                                    "error": "data.yaml not found. Generate dataset first."}
                return
            self._train_status[company_id] = {"status": "training", "epoch": 0, "total_epochs": epochs}
            model = YOLO(base_model)
            model.train(
                data=yaml_path, epochs=int(epochs), imgsz=int(img_size),
                batch=int(batch_size), patience=int(patience),
                name="epi_detector", project=str(CompanyData.epi(company_id, "models").resolve()),
                exist_ok=True, plots=True,
            )
            best_path = str(CompanyData.epi(company_id, "models", "epi_detector", "weights", "best.pt").resolve())
            self._train_status[company_id] = {"status": "complete", "model_path": best_path}
            self._models.pop(self._model_key(company_id, "best"), None)
            logger.info(f"[Company {company_id}] Training complete: {best_path}")
        except Exception as e:
            logger.error(f"[Company {company_id}] Training error: {e}")
            self._train_status[company_id] = {"status": "error", "error": str(e)}

    # --- Dataset ---
    def generate_dataset(self, company_id: int, train_split: float = 0.8) -> dict:
        import glob
        import random
        import shutil

        ann_dir = CompanyData.epi(company_id, "annotations")
        logger.info(f"[Company {company_id}] Generate dataset - annotations dir: {ann_dir}")
        logger.info(f"[Company {company_id}] Dir exists: {ann_dir.exists()}")

        # List annotation files
        label_files = sorted(ann_dir.glob("*.txt"))
        logger.info(f"[Company {company_id}] Found {len(label_files)} .txt files")

        # Get active class mapping
        remapped, remap_table = self.get_remapped_classes(company_id)
        active_ids = set(remap_table.keys())
        logger.info(f"[Company {company_id}] Active classes: {remapped}")
        logger.info(f"[Company {company_id}] Remap table: {remap_table}")

        if not label_files:
            logger.warning(f"[Company {company_id}] No .txt files in {ann_dir}")
            # Also check photos_raw for txt files
            raw_dir = CompanyData.epi(company_id, "photos_raw")
            raw_txts = list(raw_dir.rglob("*.txt"))
            if raw_txts:
                return {"error": f"No annotations in annotations/ folder, but found {len(raw_txts)} .txt files in photos_raw/. Run the Converter (PHASE 3C) first to copy them to annotations/."}
            return {"error": "No annotation .txt files found. Annotate images first (Annotate tab), then generate the dataset."}

        # Find image+label pairs
        pairs = []
        skipped_no_image = 0
        skipped_no_active = 0

        for lbl_path in label_files:
            base = lbl_path.stem
            # Find matching image
            img_found = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                cand = ann_dir / (base + ext)
                if cand.exists():
                    img_found = cand
                    break
            if not img_found:
                skipped_no_image += 1
                continue

            # Read and filter annotations
            text = lbl_path.read_text().strip()
            if not text:
                continue
            filtered = []
            for line in text.split("\n"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        old_id = int(parts[0])
                        if old_id in active_ids:
                            parts[0] = str(remap_table[old_id])
                            filtered.append(" ".join(parts[:5]))
                    except (ValueError, IndexError):
                        continue
            if filtered:
                pairs.append((str(img_found), filtered))
            else:
                skipped_no_active += 1

        logger.info(f"[Company {company_id}] Pairs: {len(pairs)}, skipped_no_image: {skipped_no_image}, skipped_no_active_class: {skipped_no_active}")

        if not pairs:
            msg = f"No valid image+label pairs found. "
            if skipped_no_image > 0:
                msg += f"{skipped_no_image} labels have no matching image in annotations/. "
                msg += "Make sure images are copied to annotations/ (use Converter or Bulk Upload). "
            if skipped_no_active > 0:
                msg += f"{skipped_no_active} labels have no active class IDs. Check PPE Config. "
            msg += f"Active class IDs: {sorted(active_ids)}"
            return {"error": msg}

        # Clear old dataset
        for sub in ["dataset/train/images", "dataset/train/labels",
                     "dataset/valid/images", "dataset/valid/labels"]:
            d = CompanyData.epi(company_id, sub)
            d.mkdir(parents=True, exist_ok=True)
            for f in d.glob("*.*"):
                f.unlink()

        # Split
        random.shuffle(pairs)
        split_idx = int(len(pairs) * train_split)
        train_pairs = pairs[:split_idx]
        valid_pairs = pairs[split_idx:]

        # Ensure at least 1 in valid
        if not valid_pairs and len(pairs) > 1:
            valid_pairs = [train_pairs.pop()]
        elif not valid_pairs:
            valid_pairs = list(train_pairs)  # duplicate for tiny datasets

        # Copy to dataset
        train_img_dir = CompanyData.epi(company_id, "dataset", "train", "images")
        train_lbl_dir = CompanyData.epi(company_id, "dataset", "train", "labels")
        valid_img_dir = CompanyData.epi(company_id, "dataset", "valid", "images")
        valid_lbl_dir = CompanyData.epi(company_id, "dataset", "valid", "labels")

        for img_path, filtered_lines in train_pairs:
            img_name = Path(img_path).name
            lbl_name = Path(img_path).stem + ".txt"
            shutil.copy2(img_path, str(train_img_dir / img_name))
            (train_lbl_dir / lbl_name).write_text("\n".join(filtered_lines))

        for img_path, filtered_lines in valid_pairs:
            img_name = Path(img_path).name
            lbl_name = Path(img_path).stem + ".txt"
            shutil.copy2(img_path, str(valid_img_dir / img_name))
            (valid_lbl_dir / lbl_name).write_text("\n".join(filtered_lines))

        logger.info(f"[Company {company_id}] Copied {len(train_pairs)} train, {len(valid_pairs)} valid")

        # Verify files were actually written
        actual_train = len(list(train_img_dir.glob("*.*")))
        actual_valid = len(list(valid_img_dir.glob("*.*")))
        logger.info(f"[Company {company_id}] Verify: {actual_train} train images, {actual_valid} valid images on disk")

        # Write data.yaml with ABSOLUTE paths
        yaml_data = {
            "train": str(train_img_dir.resolve()),
            "val": str(valid_img_dir.resolve()),
            "nc": len(remapped),
            "names": list(remapped.values()),
        }
        yaml_path = CompanyData.epi(company_id, "dataset", "data.yaml")
        import yaml as pyyaml
        yaml_content = pyyaml.dump(yaml_data, default_flow_style=False)
        yaml_path.write_text(yaml_content)

        logger.info(f"[Company {company_id}] data.yaml written to: {yaml_path} (abs: {yaml_path.resolve()})")
        logger.info(f"[Company {company_id}] data.yaml content:\n{yaml_content}")

        # Final verification
        if not yaml_path.exists():
            logger.error(f"[Company {company_id}] CRITICAL: data.yaml NOT found after write!")
            return {"error": f"Failed to write data.yaml to {yaml_path.resolve()}"}

        return {
            "total_pairs": len(pairs),
            "train": len(train_pairs),
            "valid": len(valid_pairs),
            "classes": remapped,
            "yaml_path": str(yaml_path.resolve()),
            "train_dir": str(train_img_dir.resolve()),
            "valid_dir": str(valid_img_dir.resolve()),
        }

    def list_models(self, company_id: int) -> list:
        models = []
        root = CompanyData.epi(company_id, "models")
        if root.exists():
            for f in root.rglob("*.pt"):
                models.append({"name": f.stem, "path": str(f),
                                "size_mb": round(f.stat().st_size / 1048576, 1)})
        return models


# Singletons
epi_engine = EPIEngine()
