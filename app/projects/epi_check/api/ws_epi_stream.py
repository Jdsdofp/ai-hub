# app/projects/epi_check/api/ws_epi_stream.py
"""
WebSocket EPI Video Stream — processamento contínuo de vídeo para detecção de EPI.

Fluxo:
  1. Frontend conecta em ws://.../api/v1/epi/ws/epi-stream?company_id=N
  2. Frontend envia frames JPEG como bytes binários a ~10fps
  3. Backend processa cada frame com YOLOv8 + InsightFace na GPU
  4. Backend devolve JSON com detecções, face e conformidade por frame
  5. Frontend acumula resultados e decide por maioria em X segundos
"""

import asyncio
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from loguru import logger

from app.projects.epi_check.engine.detector import epi_engine

router = APIRouter()


class EpiStreamSession:
    def __init__(self, window_seconds: float = 3.0):
        self.window_seconds = window_seconds
        self.results: deque = deque()
        self.frame_count = 0
        self.started_at: float | None = None
        self.decided = False

    def reset(self):
        self.results.clear()
        self.frame_count = 0
        self.started_at = None
        self.decided = False

    def push(self, compliant: bool, face_detected: bool, person_code: str | None):
        now = time.time()
        if self.started_at is None:
            self.started_at = now
        self.frame_count += 1
        self.results.append((now, compliant, face_detected, person_code))
        cutoff = now - self.window_seconds
        while self.results and self.results[0][0] < cutoff:
            self.results.popleft()

    @property
    def window_elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.time() - self.started_at

    @property
    def ready_to_decide(self) -> bool:
        return self.window_elapsed >= self.window_seconds and len(self.results) > 0

    def decide(self) -> dict:
        if not self.results:
            return {"type": "decision", "access_decision": "DENIED_FACE", "compliant_frames": 0,
                    "total_frames": 0, "compliance_rate": 0.0, "face_frames": 0, "face_rate": 0.0,
                    "person_code": None, "person_name": None}

        total = len(self.results)
        compliant_count = sum(1 for _, c, _, _ in self.results if c)
        face_count = sum(1 for _, _, f, _ in self.results if f)
        compliance_rate = compliant_count / total
        face_rate = face_count / total

        codes = [pc for _, _, _, pc in self.results if pc]
        best_person = max(set(codes), key=codes.count) if codes else None

        if face_rate < 0.5:
            access_decision = "DENIED_FACE"
        elif compliance_rate < 0.5:
            access_decision = "DENIED_EPI"
        else:
            access_decision = "GRANTED"

        return {"type": "decision", "access_decision": access_decision,
                "compliant_frames": compliant_count, "total_frames": total,
                "compliance_rate": round(compliance_rate, 4), "face_frames": face_count,
                "face_rate": round(face_rate, 4), "person_code": best_person,
                "person_name": None, "window_seconds": self.window_seconds}


@router.websocket("/ws/epi-stream")
async def epi_video_stream(
    websocket: WebSocket,
    company_id: int = Query(...),
    window_seconds: float = Query(3.0),
    fps: float = Query(10.0),
    confidence: float = Query(0.4),
    face_threshold: float = Query(0.45),
    detect_faces: bool = Query(True),
):
    await websocket.accept()
    client = websocket.client.host if websocket.client else "unknown"
    logger.info(f"[EpiStream] Conectado — company={company_id} client={client} window={window_seconds}s fps={fps}")

    session = EpiStreamSession(window_seconds=window_seconds)
    frame_id = 0
    best_frame = {"img": None, "score": -1.0, "detections": [], "missing": []}
    last_frame_time = 0.0
    min_interval = 1.0 / fps

    params = {"window_seconds": window_seconds, "fps": fps, "confidence": confidence,
              "face_threshold": face_threshold, "detect_faces": detect_faces}

    try:
        await websocket.send_text(json.dumps({
            "type": "connected", "company_id": company_id,
            "window_seconds": params["window_seconds"], "fps": params["fps"],
        }))

        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
                continue

            # ── Comando JSON ──────────────────────────────────────────────
            if "text" in msg and msg["text"]:
                try:
                    cmd = json.loads(msg["text"])
                    action = cmd.get("action", "")
                    if action == "start":
                        session.reset()
                        params.update({k: cmd[k] for k in ("window_seconds", "fps", "confidence",
                                       "face_threshold", "detect_faces") if k in cmd})
                        session.window_seconds = params["window_seconds"]
                        min_interval = 1.0 / params["fps"]
                        await websocket.send_text(json.dumps({"type": "started", **params}))
                    elif action == "reset":
                        session.reset()
                        frame_id = 0
                        await websocket.send_text(json.dumps({"type": "reset_ok"}))
                    elif action == "stop":
                        if session.frame_count > 0:
                            await websocket.send_text(json.dumps(session.decide()))
                        break
                    elif action == "decide_now":
                        if session.frame_count > 0:
                            await websocket.send_text(json.dumps(session.decide()))
                except json.JSONDecodeError:
                    pass
                continue

            # ── Frame JPEG (bytes) ────────────────────────────────────────
            if "bytes" not in msg or not msg["bytes"]:
                continue

            now = time.time()
            if now - last_frame_time < min_interval:
                continue
            last_frame_time = now

            frame_bytes = msg["bytes"]
            frame_id += 1
            t0 = time.time()

            try:
                arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                import uuid; cv2.imwrite(f"/opt/vision/data/debug_{uuid.uuid4().hex[:6]}.jpg", img) if __import__("os").path.exists("/opt/vision/data/debug_enable") else None

                result = epi_engine.detect_image(
                    company_id=company_id, img=img, model_name="best",
                    confidence=params["confidence"], detect_faces=params["detect_faces"],
                    face_threshold=params["face_threshold"],
                )

                processing_ms = int((time.time() - t0) * 1000)

                faces = result.get("faces", [])
                best_face = max(faces, key=lambda f: f.get("confidence", 0)) if faces else {}
                face_detected = len(faces) > 0
                face_recognized = best_face.get("recognized", False) if best_face else False
                person_code = best_face.get("person_code") if face_recognized else None
                person_name = best_face.get("person_name") if face_recognized else None
                face_confidence = best_face.get("confidence", 0.0) if best_face else 0.0
                face_bbox = best_face.get("bbox") if best_face else None

                compliant = result.get("compliant", False)
                missing = result.get("missing", [])
                detections = result.get("detections", [])

                logger.info(f"[EPI] detectado={[d['class_name'] for d in detections]} faltando={missing}")
                # Calcula score do frame (média de confiança das detecções)
                frame_score = sum(d["confidence"] for d in detections) / max(len(detections), 1) if detections else 0.0
                if frame_score > best_frame["score"]:
                    best_frame = {"img": img.copy(), "score": frame_score, "detections": detections, "missing": missing}
                session.push(compliant, face_detected, person_code)

                results_list = list(session.results)
                total_in_window = len(results_list)

                frame_result = {
                    "type": "frame_result", "frame_id": frame_id,
                    "processing_ms": processing_ms, "compliant": compliant,
                    "missing": missing, "detections": detections,
                    "face_detected": face_detected, "face_recognized": face_recognized,
                    "face_person_code": person_code,
                    "face_person_name": person_name,
                    "face_confidence": round(face_confidence, 4), "face_bbox": face_bbox,
                    "window_progress": min(session.window_elapsed / session.window_seconds, 1.0),
                    "window_elapsed": round(session.window_elapsed, 2),
                    "window_seconds": session.window_seconds,
                    "session_frame_count": session.frame_count,
                    "session_compliant_rate": round(
                        sum(1 for _, c, _, _ in results_list if c) / max(total_in_window, 1), 4),
                    "session_face_rate": round(
                        sum(1 for _, _, f, _ in results_list if f) / max(total_in_window, 1), 4),
                }

                await websocket.send_text(json.dumps(frame_result))

                if session.ready_to_decide and not session.decided:
                    session.decided = True
                    decision = session.decide()
                    # Salva melhor frame para active learning
                    if best_frame["img"] is not None and best_frame["score"] >= 0.3:
                        try:
                            review_dir = Path(f"/opt/vision/data/{company_id}/epi_check/review/images")
                            review_dir.mkdir(parents=True, exist_ok=True)
                            import uuid as _uuid
                            fname = f"{_uuid.uuid4().hex[:12]}.jpg"
                            cv2.imwrite(str(review_dir / fname), best_frame["img"])
                            # Anotação automática YOLO
                            label_dir = Path(f"/opt/vision/data/{company_id}/epi_check/review/labels")
                            label_dir.mkdir(parents=True, exist_ok=True)
                            h, w = best_frame["img"].shape[:2]
                            lines = []
                            config = epi_engine.get_ppe_config(company_id)
                            class_map = {v: k for k, v in epi_engine._load_model(company_id).names.items()}
                            for d in best_frame["detections"]:
                                cls_id = class_map.get(d["class_name"])
                                if cls_id is None: continue
                                b = d["bbox"]
                                cx = (b["x"] + b["w"] / 2) / w
                                cy = (b["y"] + b["h"] / 2) / h
                                bw = b["w"] / w
                                bh = b["h"] / h
                                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                            (label_dir / fname.replace(".jpg", ".txt")).write_text("\n".join(lines))
                            logger.info(f"[ActiveLearning] Frame salvo: {fname} score={best_frame['score']:.2f} deteccoes={len(best_frame['detections'])}")
                            # Verifica threshold para auto-retreino
                            approved = len(list(Path(f"/opt/vision/data/{company_id}/epi_check/annotations").glob("*.txt")))
                            review_count = len(list(label_dir.glob("*.txt")))
                            if review_count >= 20:
                                logger.info(f"[ActiveLearning] {review_count} frames acumulados — retreino disponivel")
                        except Exception as _e:
                            logger.warning(f"[ActiveLearning] Erro ao salvar frame: {_e}")
                    best_frame = {"img": None, "score": -1.0, "detections": [], "missing": []}
                    logger.info(f"[EpiStream] company={company_id} decision={decision['access_decision']} "
                                f"compliance={decision['compliance_rate']:.0%} face={decision['face_rate']:.0%} "
                                f"frames={decision['total_frames']}")
                    await websocket.send_text(json.dumps(decision))
                    session.reset()

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"[EpiStream] Frame {frame_id} error: {e}")
                try:
                    await websocket.send_text(json.dumps({"type": "error", "frame_id": frame_id, "message": str(e)}))
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.info(f"[EpiStream] Desconectado — company={company_id} frames={frame_id}")
    except Exception as e:
        logger.error(f"[EpiStream] Erro fatal — company={company_id}: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        logger.info(f"[EpiStream] Sessão encerrada — company={company_id} total_frames={frame_id}")
