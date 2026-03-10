## app/core/repository.py
# # Singleton — importe nos outros módulos com:
# #   from app.core.repository import repo
# repo = VisionRepository()

"""
SmartX Vision Platform v3 — MySQL Repository Layer
====================================================
Mapeado diretamente do DDL real do banco ai_hub.

Tabelas vision_*    → escritas pelo AI Hub (este sistema)
Tabelas visionapp_* → escritas/lidas pelo SmartX Hub (Node-RED/Hub)

Este arquivo cobre:
  ESCRITA  → vision_detection_events, vision_epi_detections,
             vision_alerts, vision_people, vision_face_photos,
             vision_ppe_config, vision_training_runs, vision_models,
             vision_stream_sessions, vision_snapshots, vision_datasets,
             vision_compliance_hourly, vision_compliance_daily, vision_sync_log,
             vision_annotations, vision_edge_devices

  LEITURA  → visionapp_ppe_config, visionapp_people,
             visionapp_recognition_events (dados gerenciados pelo Hub)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

from app.core.database import db


class VisionRepository:

    # =========================================================================
    # vision_detection_events
    # =========================================================================

    async def save_detection(
        self,
        company_id: int,
        result: dict,
        camera_id: Optional[int] = None,
        zone_id: Optional[int] = None,
        snapshot_path: Optional[str] = None,
        edge_device_id: Optional[str] = None,
        model_name: Optional[str] = None,
        confidence_threshold: float = 0.4,
        source_type: str = "upload",
    ) -> Optional[int]:
        try:
            detections = result.get("detections", [])
            missing    = result.get("missing", [])
            faces      = result.get("faces", [])
            compliant  = bool(result.get("compliant", False))

            epi_required = result.get("epi_required_count", len(detections))
            epi_detected = result.get("epi_detected_count", len([d for d in detections if d.get("detected", True)]))
            epi_missing  = len(missing)
            score        = round(epi_detected / epi_required, 4) if epi_required else 1.0
            sync_priority = 1 if not compliant else 5

            source_type = str(source_type or "upload")[:50].strip()
            source_type = source_type.replace('\n', ' ').replace('\r', ' ')

            person_code = person_name = None
            if faces:
                best = max(faces, key=lambda f: f.get("confidence", 0))
                if best.get("recognized"):
                    person_code = best.get("person_code")
                    person_name = best.get("person_name")



            event_id = await db.insert_get_id(
                """
                INSERT INTO vision_detection_events (
                    company_id, compliant,
                    epi_required_count, epi_detected_count, epi_missing_count,
                    compliance_score, missing_items, detections, faces,
                    snapshot_path, person_code, person_name,
                    camera_id, zone_id, edge_device_id,
                    model_name, confidence_threshold, processing_ms,
                    source_type, sync_priority
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    company_id, compliant,
                    epi_required, epi_detected, epi_missing,
                    score,
                    json.dumps(missing),
                    json.dumps(detections),
                    json.dumps(faces),
                    snapshot_path, person_code, person_name,
                    camera_id, zone_id, edge_device_id,
                    model_name or result.get("model_name"),
                    confidence_threshold,
                    result.get("processing_ms", 0),
                    source_type, sync_priority,
                ),
            )

            if event_id and detections:
                await self._save_epi_detections(company_id, event_id, detections, missing)

            return event_id

        except Exception as e:
            logger.error(f"[Repo] save_detection: {e}")
            return None

    async def _save_epi_detections(
        self, company_id: int, event_id: int, detections: list, missing: list
    ):
        try:
            classes: dict = {}
            for det in detections:
                cn = det.get("class_name", "unknown")
                if cn not in classes:
                    classes[cn] = {"instances": [], "required": det.get("quantity_required", 1)}
                classes[cn]["instances"].append(det)

            for class_name, data in classes.items():
                instances  = data["instances"]
                qty_req    = data["required"]
                qty_det    = len(instances)
                qty_miss   = max(0, qty_req - qty_det)
                compliant  = qty_det >= qty_req
                confs      = [i.get("confidence", 0) for i in instances if i.get("confidence")]
                best_conf  = max(confs) if confs else None
                avg_conf   = round(sum(confs) / len(confs), 4) if confs else None

                await db.execute(
                    """
                    INSERT INTO vision_epi_detections (
                        company_id, event_id, class_name,
                        quantity_required, quantity_detected, quantity_missing,
                        class_compliant, best_confidence, avg_confidence, all_instances
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        company_id, event_id, class_name,
                        qty_req, qty_det, qty_miss,
                        compliant, best_conf, avg_conf,
                        json.dumps(instances),
                    ),
                )
        except Exception as e:
            logger.warning(f"[Repo] _save_epi_detections (não crítico): {e}")

    async def get_recent_detections(
        self, company_id: int, limit: int = 50, only_noncompliant: bool = False
    ) -> list:
        try:
            extra = "AND compliant = 0" if only_noncompliant else ""
            return await db.fetch_all(
                f"""
                SELECT id, compliant, epi_required_count, epi_detected_count,
                       epi_missing_count, compliance_score, missing_items,
                       detections, faces, snapshot_path, person_code, person_name,
                       camera_id, zone_id, model_name, processing_ms,
                       source_type, sync_status, created_at
                FROM vision_detection_events
                WHERE company_id = %s {extra}
                ORDER BY created_at DESC LIMIT %s
                """,
                (company_id, limit),
            )
        except Exception as e:
            logger.error(f"[Repo] get_recent_detections: {e}")
            return []

    # =========================================================================
    # vision_alerts
    # =========================================================================

    async def save_alert(
        self,
        company_id: int,
        alert_type: str,
        details: dict,
        severity: str = "medium",
        person_code: Optional[str] = None,
        camera_id: Optional[int] = None,
        zone_id: Optional[int] = None,
    ) -> Optional[int]:
        try:
            return await db.insert_get_id(
                """
                INSERT INTO vision_alerts
                    (company_id, alert_type, severity, details,
                     person_code, camera_id, zone_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                """,
                (company_id, alert_type, severity, json.dumps(details),
                 person_code, camera_id, zone_id),
            )
        except Exception as e:
            logger.error(f"[Repo] save_alert: {e}")
            return None

    async def get_alerts(
        self, company_id: int, limit: int = 50, unresolved_only: bool = False
    ) -> list:
        try:
            extra = "AND resolved = 0" if unresolved_only else ""
            return await db.fetch_all(
                f"""
                SELECT id, alert_type, severity, details, person_code,
                       camera_id, zone_id, acknowledged, resolved,
                       sync_status, created_at
                FROM vision_alerts
                WHERE company_id = %s {extra}
                ORDER BY created_at DESC LIMIT %s
                """,
                (company_id, limit),
            )
        except Exception as e:
            logger.error(f"[Repo] get_alerts: {e}")
            return []

    # =========================================================================
    # vision_people  +  vision_face_photos
    # =========================================================================

    # async def upsert_person(
    #     self,
    #     company_id: int,
    #     person_code: str,
    #     person_name: str,
    #     badge_id: str = "",
    #     department: str = "",
    # ) -> Optional[int]:
    #     try:
    #         return await db.insert_get_id(
    #             """
    #             INSERT INTO vision_people
    #                 (company_id, person_code, person_name, badge_id, department)
    #             VALUES (%s,%s,%s,%s,%s)
    #             ON DUPLICATE KEY UPDATE
    #                 person_name  = VALUES(person_name),
    #                 badge_id     = VALUES(badge_id),
    #                 department   = VALUES(department),
    #                 updated_at   = CURRENT_TIMESTAMP
    #             """,
    #             (company_id, person_code, person_name, badge_id, department),
    #         )
    #     except Exception as e:
    #         logger.error(f"[Repo] upsert_person: {e}")
    #         return None

    async def upsert_person(
        self,
        company_id: int,
        person_code: str,
        person_name: str,
        badge_id: str = "",
        department: str = "",
    ) -> Optional[int]:
        try:
            await db.execute(
                """
                INSERT INTO vision_people
                    (company_id, person_code, person_name, badge_id, department)
                VALUES (%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                    person_name  = VALUES(person_name),
                    badge_id     = VALUES(badge_id),
                    department   = VALUES(department),
                    updated_at   = CURRENT_TIMESTAMP
                """,
                (company_id, person_code, person_name, badge_id, department),
            )
            # ON DUPLICATE KEY retorna lastrowid=0 — busca o ID real
            row = await db.fetch_one(
                "SELECT id FROM vision_people WHERE company_id=%s AND person_code=%s",
                (company_id, person_code),
            )
            return row["id"] if row else None
        except Exception as e:
            logger.error(f"[Repo] upsert_person: {e}")
            return None

    async def save_face_photo(
        self,
        company_id: int,
        person_code: str,
        filename: str,
        filepath: str,
        quality_score: Optional[float] = None,
        embedding: Optional[list] = None,
    ) -> Optional[int]:
        try:
            photo_id = await db.insert_get_id(
                """
                INSERT INTO vision_face_photos
                    (company_id, person_code, filename, filepath,
                     quality_score, embedding)
                VALUES (%s,%s,%s,%s,%s,%s)
                """,
                (
                    company_id, person_code, filename, filepath,
                    quality_score,
                    json.dumps(embedding) if embedding else None,
                ),
            )
            if photo_id:
                await db.execute(
                    """
                    UPDATE vision_people
                    SET face_photos_count = face_photos_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE company_id = %s AND person_code = %s
                    """,
                    (company_id, person_code),
                )
            return photo_id
        except Exception as e:
            logger.error(f"[Repo] save_face_photo: {e}")
            return None

    async def list_people(self, company_id: int, active_only: bool = True) -> list:
        try:
            extra = "AND active = 1" if active_only else ""
            return await db.fetch_all(
                f"""
                SELECT person_code, person_name, badge_id, department,
                       face_photos_count, active,
                       last_entry_at, last_exit_at, is_inside, created_at
                FROM vision_people
                WHERE company_id = %s {extra}
                ORDER BY person_name
                """,
                (company_id,),
            )
        except Exception as e:
            logger.error(f"[Repo] list_people: {e}")
            return []

    async def get_person(self, company_id: int, person_code: str) -> Optional[dict]:
        try:
            return await db.fetch_one(
                """
                SELECT person_code, person_name, badge_id, department,
                       face_photos_count, active, is_inside,
                       last_entry_at, last_exit_at,
                       compliance_mode, photo_count_for_validation
                FROM vision_people
                WHERE company_id = %s AND person_code = %s
                """,
                (company_id, person_code),
            )
        except Exception as e:
            logger.error(f"[Repo] get_person: {e}")
            return None

    async def get_face_photos(self, company_id: int, person_code: str) -> list:
        try:
            return await db.fetch_all(
                """
                SELECT id, filename, filepath, quality_score, registered_at
                FROM vision_face_photos
                WHERE company_id = %s AND person_code = %s
                ORDER BY registered_at ASC
                """,
                (company_id, person_code),
            )
        except Exception as e:
            logger.error(f"[Repo] get_face_photos: {e}")
            return []

    # =========================================================================
    # vision_ppe_config
    # =========================================================================

    async def save_ppe_config(self, company_id: int, config: dict) -> bool:
        try:
            for class_name, cfg in config.items():
                if isinstance(cfg, bool):
                    cfg = {"enabled": cfg}
                await db.execute(
                    """
                    INSERT INTO vision_ppe_config
                        (company_id, class_name, enabled, required,
                         min_quantity, confidence_min, display_name)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                        enabled        = VALUES(enabled),
                        required       = VALUES(required),
                        min_quantity   = VALUES(min_quantity),
                        confidence_min = VALUES(confidence_min),
                        display_name   = VALUES(display_name),
                        updated_at     = CURRENT_TIMESTAMP
                    """,
                    (
                        company_id, class_name,
                        cfg.get("enabled", True),
                        cfg.get("required", True),
                        cfg.get("min_quantity", 1),
                        cfg.get("confidence_min", 0.4),
                        cfg.get("display_name"),
                    ),
                )
            return True
        except Exception as e:
            logger.error(f"[Repo] save_ppe_config: {e}")
            return False

    async def get_ppe_config(self, company_id: int) -> Optional[dict]:
        try:
            for table in ("vision_ppe_config", "visionapp_ppe_config"):
                rows = await db.fetch_all(
                    f"""
                    SELECT class_name, enabled, required, min_quantity,
                           max_quantity, confidence_min, display_name, body_region
                    FROM {table}
                    WHERE company_id = %s
                    """,
                    (company_id,),
                )
                if rows:
                    return {
                        r["class_name"]: {
                            "enabled":        bool(r["enabled"]),
                            "required":       bool(r["required"]),
                            "min_quantity":   r["min_quantity"],
                            "max_quantity":   r.get("max_quantity"),
                            "confidence_min": r["confidence_min"],
                            "display_name":   r.get("display_name"),
                            "body_region":    r.get("body_region"),
                        }
                        for r in rows
                    }
            return None
        except Exception as e:
            logger.error(f"[Repo] get_ppe_config: {e}")
            return None

    # =========================================================================
    # vision_training_runs
    # =========================================================================

    async def create_training_run(
        self,
        company_id: int,
        base_model: str,
        epochs: int,
        batch_size: int,
        img_size: int,
        classes: dict,
        dataset_id: Optional[int] = None,
    ) -> Optional[int]:
        try:
            return await db.insert_get_id(
                """
                INSERT INTO vision_training_runs
                    (company_id, dataset_id, base_model, epochs,
                     batch_size, img_size, classes, status, started_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,'training', NOW())
                """,
                (company_id, dataset_id, base_model, epochs,
                 batch_size, img_size, json.dumps(classes)),
            )
        except Exception as e:
            logger.error(f"[Repo] create_training_run: {e}")
            return None

    async def update_training_run(
        self,
        run_id: int,
        status: str,
        best_map50: Optional[float] = None,
        best_map50_95: Optional[float] = None,
        model_path: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        try:
            set_completed = "completed_at = NOW()," if status in ("complete", "error") else ""
            await db.execute(
                f"""
                UPDATE vision_training_runs SET
                    status        = %s,
                    best_map50    = COALESCE(%s, best_map50),
                    best_map50_95 = COALESCE(%s, best_map50_95),
                    model_path    = COALESCE(%s, model_path),
                    error_message = COALESCE(%s, error_message)
                    {(',' + set_completed).rstrip(',')}
                WHERE id = %s
                """,
                (status, best_map50, best_map50_95, model_path, error_message, run_id),
            )
            return True
        except Exception as e:
            logger.error(f"[Repo] update_training_run: {e}")
            return False

    async def get_training_history(self, company_id: int, limit: int = 20) -> list:
        try:
            return await db.fetch_all(
                """
                SELECT id, dataset_id, base_model, epochs, batch_size,
                       img_size, classes, status,
                       best_map50, best_map50_95, model_path,
                       error_message, started_at, completed_at, created_at
                FROM vision_training_runs
                WHERE company_id = %s
                ORDER BY created_at DESC LIMIT %s
                """,
                (company_id, limit),
            )
        except Exception as e:
            logger.error(f"[Repo] get_training_history: {e}")
            return []

    # =========================================================================
    # vision_models  — save_model corrigido (v3.3)
    # =========================================================================

    async def save_model(
        self,
        company_id: int,
        model_name: str,
        filepath: str,
        base_model: Optional[str] = None,
        training_run_id: Optional[int] = None,
        map50: Optional[float] = None,
        map50_95: Optional[float] = None,
        classes: Optional[dict] = None,
    ) -> Optional[int]:
        """
        Calcula filename e file_size_mb automaticamente do filepath.
        ON DUPLICATE KEY usa COALESCE para não sobrescrever métricas com NULL.
        """
        try:
            p            = Path(filepath)
            filename     = p.name
            file_size_mb = round(p.stat().st_size / (1024 * 1024), 2) if p.exists() else None

            return await db.insert_get_id(
                """
                INSERT INTO vision_models
                    (company_id, model_name, filename, filepath, file_size_mb,
                     base_model, classes, map50, map50_95, training_run, active)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, 1)
                ON DUPLICATE KEY UPDATE
                    filename     = VALUES(filename),
                    filepath     = VALUES(filepath),
                    file_size_mb = VALUES(file_size_mb),
                    base_model   = COALESCE(VALUES(base_model),   base_model),
                    classes      = COALESCE(VALUES(classes),      classes),
                    map50        = COALESCE(VALUES(map50),        map50),
                    map50_95     = COALESCE(VALUES(map50_95),     map50_95),
                    training_run = COALESCE(VALUES(training_run), training_run),
                    active       = 1
                """,
                (
                    company_id, model_name, filename, filepath, file_size_mb,
                    base_model,
                    json.dumps(classes) if isinstance(classes, dict) else classes,
                    map50, map50_95, training_run_id,
                ),
            )
        except Exception as e:
            logger.error(f"[Repo] save_model: {e}")
            return None

    async def list_models(self, company_id: int, active_only: bool = True) -> list:
        try:
            extra = "AND active = 1" if active_only else ""
            return await db.fetch_all(
                f"""
                SELECT id, model_name, filename, filepath, file_size_mb,
                       base_model, classes, map50, map50_95,
                       training_run, active, sync_status, created_at
                FROM vision_models
                WHERE company_id = %s {extra}
                ORDER BY created_at DESC
                """,
                (company_id,),
            )
        except Exception as e:
            logger.error(f"[Repo] list_models: {e}")
            return []

    # =========================================================================
    # vision_stream_sessions
    # =========================================================================

    async def save_stream_session(
        self,
        company_id: int,
        session_id: str,
        source_url: str,
        source_type: str,
        model_name: str = "best",
        confidence: float = 0.4,
        detect_faces: bool = False,
    ) -> bool:
        try:
            await db.execute(
                """
                INSERT IGNORE INTO vision_stream_sessions
                    (company_id, session_id, source_url, source_type,
                     model_name, confidence, detect_faces, status)
                VALUES (%s,%s,%s,%s,%s,%s,%s,'active')
                """,
                (company_id, session_id, source_url, source_type,
                 model_name, confidence, detect_faces),
            )
            return True
        except Exception as e:
            logger.warning(f"[Repo] save_stream_session: {e}")
            return False

    async def close_stream_session(
        self,
        session_id: str,
        frame_count: int = 0,
        avg_fps: float = 0,
        compliant_pct: Optional[float] = None,
        status: str = "stopped",
        error_message: Optional[str] = None,
    ) -> bool:
        try:
            await db.execute(
                """
                UPDATE vision_stream_sessions SET
                    frame_count   = %s,
                    avg_fps       = %s,
                    compliant_pct = COALESCE(%s, compliant_pct),
                    status        = %s,
                    error_message = COALESCE(%s, error_message),
                    stopped_at    = NOW()
                WHERE session_id = %s
                """,
                (frame_count, avg_fps, compliant_pct, status, error_message, session_id),
            )
            return True
        except Exception as e:
            logger.warning(f"[Repo] close_stream_session: {e}")
            return False

    # =========================================================================
    # vision_snapshots
    # =========================================================================

    async def save_snapshot(
        self,
        company_id: int,
        filename: str,
        filepath: str,
        snapshot_type: str = "EPI_DETECTION",
        source_type: str = "stream",
        event_id: Optional[int] = None,
        session_id: Optional[int] = None,
        file_size_kb: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Optional[int]:
        try:
            return await db.insert_get_id(
                """
                INSERT INTO vision_snapshots
                    (company_id, event_id, session_id, filename, filepath,
                     file_size_kb, width, height, snapshot_type, source_type)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (company_id, event_id, session_id, filename, filepath,
                 file_size_kb, width, height, snapshot_type, source_type),
            )
        except Exception as e:
            logger.warning(f"[Repo] save_snapshot: {e}")
            return None

    # =========================================================================
    # vision_datasets
    # =========================================================================

    async def save_dataset(
        self,
        company_id: int,
        train_count: int,
        valid_count: int,
        classes: dict,
        yaml_path: str,
        train_split: float = 0.8,
        valid_split: float = 0.15,
    ) -> Optional[int]:
        try:
            return await db.insert_get_id(
                """
                INSERT INTO vision_datasets
                    (company_id, train_count, valid_count, total_pairs,
                     classes, train_split, valid_split, yaml_path)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    company_id, train_count, valid_count,
                    train_count + valid_count,
                    json.dumps(classes),
                    train_split, valid_split, yaml_path,
                ),
            )
        except Exception as e:
            logger.error(f"[Repo] save_dataset: {e}")
            return None

    # =========================================================================
    # vision_compliance_hourly / vision_compliance_daily
    # =========================================================================

    async def upsert_compliance_hourly(
        self,
        company_id: int,
        hour_ts: str,
        total: int,
        compliant: int,
        zone_id: Optional[int] = None,
    ) -> bool:
        try:
            rate = round(compliant / total, 4) if total else 0.0
            await db.execute(
                """
                INSERT INTO vision_compliance_hourly
                    (company_id, hour_ts, zone_id,
                     total_sessions, compliant_count, compliance_rate)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                    total_sessions  = total_sessions  + VALUES(total_sessions),
                    compliant_count = compliant_count + VALUES(compliant_count),
                    compliance_rate = compliant_count / total_sessions
                """,
                (company_id, hour_ts, zone_id, total, compliant, rate),
            )
            return True
        except Exception as e:
            logger.warning(f"[Repo] upsert_compliance_hourly: {e}")
            return False

    async def upsert_compliance_daily(
        self,
        company_id: int,
        date: str,
        total: int,
        compliant: int,
        zone_id: Optional[int] = None,
    ) -> bool:
        try:
            rate = round(compliant / total, 4) if total else 0.0
            await db.execute(
                """
                INSERT INTO vision_compliance_daily
                    (company_id, date, zone_id,
                     total_sessions, compliant_count, compliance_rate)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                    total_sessions  = total_sessions  + VALUES(total_sessions),
                    compliant_count = compliant_count + VALUES(compliant_count),
                    compliance_rate = compliant_count / total_sessions
                """,
                (company_id, date, zone_id, total, compliant, rate),
            )
            return True
        except Exception as e:
            logger.warning(f"[Repo] upsert_compliance_daily: {e}")
            return False

    # =========================================================================
    # vision_sync_log
    # =========================================================================

    async def log_sync(
        self,
        entity_type: str,
        entity_id: int,
        direction: str,
        status: str,
        http_status: Optional[int] = None,
        duration_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        hub_entity_id: Optional[int] = None,
        batch_id: Optional[str] = None,
    ) -> bool:
        try:
            await db.execute(
                """
                INSERT INTO vision_sync_log
                    (sync_direction, entity_type, entity_id, hub_entity_id,
                     status, http_status, duration_ms, error_message, batch_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (direction, entity_type, entity_id, hub_entity_id,
                 status, http_status, duration_ms, error_message, batch_id),
            )
            return True
        except Exception as e:
            logger.warning(f"[Repo] log_sync: {e}")
            return False

    # =========================================================================
    # vision_annotations
    # =========================================================================

    async def save_annotations(
        self,
        company_id: int,
        image_name: str,
        annotations: list,
        active_classes: dict,
        source: str = "manual",
    ) -> bool:
        try:
            await db.execute(
                "DELETE FROM vision_annotations WHERE company_id = %s AND image_name = %s",
                (company_id, image_name)
            )
            for ann in annotations:
                cid = int(ann.get('class_id', ann.get('classId', 0)))
                await db.execute(
                    """
                    INSERT INTO vision_annotations
                        (company_id, image_name, class_id, class_name, cx, cy, w, h, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        company_id, image_name, cid,
                        active_classes.get(cid, f"class_{cid}"),
                        float(ann.get('cx', 0)),
                        float(ann.get('cy', 0)),
                        float(ann.get('w', 0)),
                        float(ann.get('h', 0)),
                        source,
                    )
                )
            return True
        except Exception as e:
            logger.error(f"[Repo] save_annotations: {e}")
            return False

    async def get_annotations(self, company_id: int, image_name: str) -> list:
        try:
            return await db.fetch_all(
                """
                SELECT id, class_id, class_name, cx, cy, w, h, source, created_at
                FROM vision_annotations
                WHERE company_id = %s AND image_name = %s
                ORDER BY id ASC
                """,
                (company_id, image_name),
            )
        except Exception as e:
            logger.error(f"[Repo] get_annotations: {e}")
            return []

    async def get_annotation_stats(self, company_id: int) -> dict:
        try:
            images = await db.fetch_one(
                """
                SELECT COUNT(DISTINCT image_name) AS total_images,
                       COUNT(*) AS total_annotations
                FROM vision_annotations
                WHERE company_id = %s
                """,
                (company_id,),
            )
            classes = await db.fetch_all(
                """
                SELECT class_name, COUNT(*) AS count
                FROM vision_annotations
                WHERE company_id = %s
                GROUP BY class_name
                ORDER BY count DESC
                """,
                (company_id,),
            )
            return {
                "total_images":      (images or {}).get("total_images", 0),
                "total_annotations": (images or {}).get("total_annotations", 0),
                "by_class":          {r["class_name"]: r["count"] for r in classes},
            }
        except Exception as e:
            logger.error(f"[Repo] get_annotation_stats: {e}")
            return {"total_images": 0, "total_annotations": 0, "by_class": {}}

    # =========================================================================
    # vision_training_photos
    # =========================================================================

    async def save_training_photo(
        self,
        company_id: int,
        category: str,
        filename: str,
        filepath: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        file_size_kb: Optional[int] = None,
        has_label: bool = False,
        upload_type: str = "single",  # single | bulk | augmented | industrial
    ) -> Optional[int]:
        try:
            return await db.insert_get_id(
                """
                INSERT INTO vision_training_photos
                    (company_id, category, filename, filepath,
                     width, height, file_size_kb, has_label, upload_type)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (company_id, category, filename, filepath,
                 width, height, file_size_kb, has_label, upload_type),
            )
        except Exception as e:
            logger.error(f"[Repo] save_training_photo: {e}")
            return None

    async def update_training_photo_label(
        self, company_id: int, filename: str, has_label: bool = True
    ) -> bool:
        """Atualiza has_label quando um .txt for salvo para a foto."""
        try:
            await db.execute(
                """
                UPDATE vision_training_photos
                SET has_label = %s
                WHERE company_id = %s AND filename = %s
                """,
                (has_label, company_id, filename),
            )
            return True
        except Exception as e:
            logger.error(f"[Repo] update_training_photo_label: {e}")
            return False

    async def get_training_photos_stats(self, company_id: int) -> dict:
        """Resumo de fotos de treino por categoria."""
        try:
            rows = await db.fetch_all(
                """
                SELECT category,
                       COUNT(*) AS total,
                       SUM(has_label) AS labeled
                FROM vision_training_photos
                WHERE company_id = %s
                GROUP BY category
                ORDER BY category
                """,
                (company_id,),
            )
            return {r["category"]: {"total": r["total"], "labeled": r["labeled"]} for r in rows}
        except Exception as e:
            logger.error(f"[Repo] get_training_photos_stats: {e}")
            return {}

    # =========================================================================
    # vision_edge_devices
    # =========================================================================

    async def register_edge_device(
        self,
        company_id: int,
        device_code: str,
        device_name: Optional[str] = None,
        site_id: Optional[int] = None,
        zone_id: Optional[int] = None,
        api_base_url: Optional[str] = None,
    ) -> Optional[int]:
        """Registra ou atualiza um dispositivo edge. Retorna o id."""
        try:
            await db.execute(
                """
                INSERT INTO vision_edge_devices
                    (company_id, device_code, device_name, site_id, zone_id,
                     api_base_url, active, last_heartbeat)
                VALUES (%s, %s, %s, %s, %s, %s, 1, NOW())
                ON DUPLICATE KEY UPDATE
                    device_name    = COALESCE(VALUES(device_name),  device_name),
                    site_id        = COALESCE(VALUES(site_id),      site_id),
                    zone_id        = COALESCE(VALUES(zone_id),      zone_id),
                    api_base_url   = COALESCE(VALUES(api_base_url), api_base_url),
                    active         = 1,
                    last_heartbeat = NOW()
                """,
                (company_id, device_code, device_name, site_id, zone_id, api_base_url),
            )
            row = await db.fetch_one(
                "SELECT id FROM vision_edge_devices WHERE company_id=%s AND device_code=%s",
                (company_id, device_code),
            )
            return row["id"] if row else None
        except Exception as e:
            logger.error(f"[Repo] register_edge_device: {e}")
            return None

    async def heartbeat_edge_device(self, company_id: int, device_code: str) -> bool:
        """Atualiza last_heartbeat do dispositivo."""
        try:
            await db.execute(
                """
                UPDATE vision_edge_devices
                SET last_heartbeat = NOW()
                WHERE company_id = %s AND device_code = %s
                """,
                (company_id, device_code),
            )
            return True
        except Exception as e:
            logger.error(f"[Repo] heartbeat_edge_device: {e}")
            return False

    async def list_edge_devices(
        self, company_id: int, active_only: bool = True
    ) -> list:
        """Lista dispositivos com campo calculado online (heartbeat < 2 min)."""
        try:
            sql    = "SELECT * FROM vision_edge_devices WHERE company_id = %s"
            params = [company_id]
            if active_only:
                sql += " AND active = 1"
            sql += " ORDER BY device_name"
            rows = await db.fetch_all(sql, params)
            now  = datetime.utcnow()
            result = []
            for r in rows:
                d  = dict(r)
                hb = d.get("last_heartbeat")
                if hb:
                    delta                  = (now - hb).total_seconds()
                    d["online"]            = delta < 120
                    d["last_seen_seconds"] = int(delta)
                else:
                    d["online"]            = False
                    d["last_seen_seconds"] = None
                result.append(d)
            return result
        except Exception as e:
            logger.error(f"[Repo] list_edge_devices: {e}")
            return []

    async def deactivate_edge_device(
        self, company_id: int, device_code: str
    ) -> bool:
        """Soft delete — marca active = 0."""
        try:
            await db.execute(
                """
                UPDATE vision_edge_devices SET active = 0
                WHERE company_id = %s AND device_code = %s
                """,
                (company_id, device_code),
            )
            return True
        except Exception as e:
            logger.error(f"[Repo] deactivate_edge_device: {e}")
            return False

    # =========================================================================
    # LEITURA visionapp_* (gerenciados pelo Hub)
    # =========================================================================

    async def get_hub_people(self, company_id: int, active_only: bool = True) -> list:
        try:
            extra = "AND active = 1" if active_only else ""
            return await db.fetch_all(
                f"""
                SELECT person_code, person_name, badge_id, department,
                       face_photos_count, is_inside, last_entry_at, last_exit_at
                FROM visionapp_people
                WHERE company_id = %s {extra}
                ORDER BY person_name
                """,
                (company_id,),
            )
        except Exception as e:
            logger.error(f"[Repo] get_hub_people: {e}")
            return []

    async def get_hub_recognition_events(
        self, company_id: int, limit: int = 50
    ) -> list:
        try:
            return await db.fetch_all(
                """
                SELECT id, event_timestamp, camera_id, zone_id,
                       person_code, person_name,
                       face_recognized, face_confidence,
                       epi_compliant, compliance_score, compliance_detail,
                       epi_units_required, epi_units_detected, epi_units_missing,
                       access_decision, processing_time_ms, created_at
                FROM visionapp_recognition_events
                WHERE company_id = %s
                ORDER BY event_timestamp DESC LIMIT %s
                """,
                (company_id, limit),
            )
        except Exception as e:
            logger.error(f"[Repo] get_hub_recognition_events: {e}")
            return []

    # =========================================================================
    # ANALYTICS / DASHBOARD
    # =========================================================================

    async def get_dashboard_stats(self, company_id: int) -> dict:
        try:
            today = await db.fetch_one(
                """
                SELECT COUNT(*) AS total,
                       SUM(compliant) AS compliant,
                       ROUND(AVG(compliant)*100,1) AS rate
                FROM vision_detection_events
                WHERE company_id = %s AND DATE(created_at) = CURDATE()
                """,
                (company_id,),
            )
            week = await db.fetch_one(
                """
                SELECT COUNT(*) AS total,
                       ROUND(AVG(compliant)*100,1) AS rate
                FROM vision_detection_events
                WHERE company_id = %s
                  AND created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                """,
                (company_id,),
            )
            people = await db.fetch_one(
                "SELECT COUNT(*) AS cnt FROM vision_people WHERE company_id=%s AND active=1",
                (company_id,),
            )
            alerts = await db.fetch_one(
                "SELECT COUNT(*) AS cnt FROM vision_alerts WHERE company_id=%s AND resolved=0",
                (company_id,),
            )
            models = await db.fetch_one(
                "SELECT COUNT(*) AS cnt FROM vision_models WHERE company_id=%s AND active=1",
                (company_id,),
            )
            return {
                "today":        today or {},
                "week":         week or {},
                "people_count": (people or {}).get("cnt", 0),
                "alerts_open":  (alerts or {}).get("cnt", 0),
                "models_count": (models or {}).get("cnt", 0),
            }
        except Exception as e:
            logger.error(f"[Repo] get_dashboard_stats: {e}")
            return {"today": {}, "week": {}, "people_count": 0, "alerts_open": 0, "models_count": 0}

    async def get_hourly_compliance(self, company_id: int, hours: int = 24) -> list:
        try:
            rows = await db.fetch_all(
                """
                SELECT hour_ts, total_sessions, compliant_count, compliance_rate
                FROM vision_compliance_hourly
                WHERE company_id = %s
                  AND hour_ts >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                ORDER BY hour_ts ASC
                """,
                (company_id, hours),
            )
            if rows:
                return rows
            return await db.fetch_all(
                """
                SELECT
                    DATE_FORMAT(created_at,'%Y-%m-%d %H:00:00') AS hour_ts,
                    COUNT(*)       AS total_sessions,
                    SUM(compliant) AS compliant_count,
                    ROUND(AVG(compliant)*100,1) AS compliance_rate
                FROM vision_detection_events
                WHERE company_id = %s
                  AND created_at >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                GROUP BY hour_ts ORDER BY hour_ts ASC
                """,
                (company_id, hours),
            )
        except Exception as e:
            logger.error(f"[Repo] get_hourly_compliance: {e}")
            return []

    async def get_missing_ppe_ranking(self, company_id: int, days: int = 7) -> list:
        try:
            rows = await db.fetch_all(
                """
                SELECT class_name, SUM(quantity_missing) AS total_missing
                FROM vision_epi_detections
                WHERE company_id = %s
                  AND quantity_missing > 0
                  AND created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                GROUP BY class_name
                ORDER BY total_missing DESC
                """,
                (company_id, days),
            )
            if rows:
                return rows
            raw = await db.fetch_all(
                """
                SELECT missing_items FROM vision_detection_events
                WHERE company_id=%s AND compliant=0
                  AND created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                """,
                (company_id, days),
            )
            counts: dict = {}
            for r in raw:
                for item in json.loads(r.get("missing_items") or "[]"):
                    counts[item] = counts.get(item, 0) + 1
            return sorted(
                [{"class_name": k, "total_missing": v} for k, v in counts.items()],
                key=lambda x: x["total_missing"], reverse=True,
            )
        except Exception as e:
            logger.error(f"[Repo] get_missing_ppe_ranking: {e}")
            return []

    async def get_compliance_summary(self, company_id: int, days: int = 7) -> dict:
        try:
            return await db.fetch_one(
                """
                SELECT
                    COUNT(*)                    AS total,
                    SUM(compliant)              AS compliant_count,
                    SUM(NOT compliant)          AS noncompliant_count,
                    ROUND(AVG(compliant)*100,1) AS compliance_rate,
                    AVG(compliance_score)       AS avg_score,
                    AVG(processing_ms)          AS avg_processing_ms
                FROM vision_detection_events
                WHERE company_id = %s
                  AND created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                """,
                (company_id, days),
            ) or {}
        except Exception as e:
            logger.error(f"[Repo] get_compliance_summary: {e}")
            return {}

    # =========================================================================
    # visionapp_access_zone_persons
    # =========================================================================

    async def check_zone_access(
        self,
        company_id: int,
        person_code: str,
        zone_id: int,
    ) -> dict:
        """
        Verifica se uma pessoa tem acesso liberado para uma zona.
        Retorna dict com:
          allowed      → bool (True = pode entrar)
          reason       → str  (GRANTED | BLOCKED | NO_PERMISSION | EXPIRED | INACTIVE)
          blocked_reason → str | None
          epi_profile_code → str | None (perfil EPI exigido para a zona)
          exposure_limit_min → float | None
          schedule_id  → int | None
        """
        try:
            row = await db.fetch_one(
                """
                SELECT access_allowed, active, blocked, blocked_reason,
                       schedule_id, valid_from, valid_until,
                       epi_profile_code, exposure_limit_min
                FROM visionapp_access_zone_persons
                WHERE company_id = %s AND person_code = %s AND zone_id = %s
                LIMIT 1
                """,
                (company_id, person_code, zone_id),
            )
            if not row:
                return {
                    "allowed": False,
                    "reason": "NO_PERMISSION",
                    "blocked_reason": None,
                    "epi_profile_code": None,
                    "exposure_limit_min": None,
                    "schedule_id": None,
                }

            if not row.get("active"):
                return {
                    "allowed": False, "reason": "INACTIVE",
                    "blocked_reason": None,
                    "epi_profile_code": row.get("epi_profile_code"),
                    "exposure_limit_min": row.get("exposure_limit_min"),
                    "schedule_id": row.get("schedule_id"),
                }

            if row.get("blocked"):
                return {
                    "allowed": False, "reason": "BLOCKED",
                    "blocked_reason": row.get("blocked_reason"),
                    "epi_profile_code": row.get("epi_profile_code"),
                    "exposure_limit_min": row.get("exposure_limit_min"),
                    "schedule_id": row.get("schedule_id"),
                }

            today = date.today()
            valid_from  = row.get("valid_from")
            valid_until = row.get("valid_until")
            if valid_from  and today < valid_from:
                return {
                    "allowed": False, "reason": "EXPIRED",
                    "blocked_reason": f"Access not valid until {valid_from}",
                    "epi_profile_code": row.get("epi_profile_code"),
                    "exposure_limit_min": row.get("exposure_limit_min"),
                    "schedule_id": row.get("schedule_id"),
                }
            if valid_until and today > valid_until:
                return {
                    "allowed": False, "reason": "EXPIRED",
                    "blocked_reason": f"Access expired on {valid_until}",
                    "epi_profile_code": row.get("epi_profile_code"),
                    "exposure_limit_min": row.get("exposure_limit_min"),
                    "schedule_id": row.get("schedule_id"),
                }

            if not row.get("access_allowed"):
                return {
                    "allowed": False, "reason": "NO_PERMISSION",
                    "blocked_reason": None,
                    "epi_profile_code": row.get("epi_profile_code"),
                    "exposure_limit_min": row.get("exposure_limit_min"),
                    "schedule_id": row.get("schedule_id"),
                }

            return {
                "allowed": True, "reason": "GRANTED",
                "blocked_reason": None,
                "epi_profile_code": row.get("epi_profile_code"),
                "exposure_limit_min": row.get("exposure_limit_min"),
                "schedule_id": row.get("schedule_id"),
            }
        except Exception as e:
            logger.error(f"[Repo] check_zone_access: {e}")
            return {"allowed": False, "reason": "ERROR", "blocked_reason": str(e),
                    "epi_profile_code": None, "exposure_limit_min": None, "schedule_id": None}

    async def upsert_zone_person(
        self,
        company_id: int,
        person_code: str,
        zone_id: int,
        person_name: Optional[str] = None,
        site_id: Optional[int] = None,
        access_allowed: bool = True,
        schedule_id: Optional[int] = None,
        valid_from: Optional[str] = None,    # "YYYY-MM-DD"
        valid_until: Optional[str] = None,   # "YYYY-MM-DD"
        epi_profile_code: Optional[str] = None,
        exposure_limit_min: Optional[float] = None,
        active: bool = True,
    ) -> bool:
        """INSERT ... ON DUPLICATE KEY UPDATE — cria ou atualiza vínculo pessoa-zona."""
        try:
            await db.execute(
                """
                INSERT INTO visionapp_access_zone_persons (
                    company_id, person_code, zone_id, person_name, site_id,
                    access_allowed, schedule_id,
                    valid_from, valid_until,
                    epi_profile_code, exposure_limit_min, active
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                    person_name        = COALESCE(VALUES(person_name), person_name),
                    site_id            = COALESCE(VALUES(site_id), site_id),
                    access_allowed     = VALUES(access_allowed),
                    schedule_id        = VALUES(schedule_id),
                    valid_from         = VALUES(valid_from),
                    valid_until        = VALUES(valid_until),
                    epi_profile_code   = VALUES(epi_profile_code),
                    exposure_limit_min = VALUES(exposure_limit_min),
                    active             = VALUES(active),
                    updated_at         = NOW()
                """,
                (
                    company_id, person_code, zone_id, person_name, site_id,
                    access_allowed, schedule_id,
                    valid_from, valid_until,
                    epi_profile_code, exposure_limit_min, active,
                ),
            )
            return True
        except Exception as e:
            logger.error(f"[Repo] upsert_zone_person: {e}")
            return False

    async def block_zone_person(
        self,
        company_id: int,
        person_code: str,
        zone_id: int,
        reason: str,
    ) -> bool:
        """Bloqueia uma pessoa em uma zona específica."""
        try:
            await db.execute(
                """
                UPDATE visionapp_access_zone_persons SET
                    blocked        = 1,
                    blocked_reason = %s,
                    blocked_at     = NOW(),
                    updated_at     = NOW()
                WHERE company_id = %s AND person_code = %s AND zone_id = %s
                """,
                (reason, company_id, person_code, zone_id),
            )
            return True
        except Exception as e:
            logger.error(f"[Repo] block_zone_person: {e}")
            return False

    async def unblock_zone_person(
        self,
        company_id: int,
        person_code: str,
        zone_id: int,
    ) -> bool:
        """Desbloqueia uma pessoa em uma zona."""
        try:
            await db.execute(
                """
                UPDATE visionapp_access_zone_persons SET
                    blocked        = 0,
                    blocked_reason = NULL,
                    blocked_at     = NULL,
                    updated_at     = NOW()
                WHERE company_id = %s AND person_code = %s AND zone_id = %s
                """,
                (company_id, person_code, zone_id),
            )
            return True
        except Exception as e:
            logger.error(f"[Repo] unblock_zone_person: {e}")
            return False

    async def list_zone_persons(
        self,
        company_id: int,
        zone_id: Optional[int] = None,
        person_code: Optional[str] = None,
        active_only: bool = True,
    ) -> list:
        """Lista vínculos pessoa-zona com filtros opcionais."""
        try:
            conditions = ["company_id = %s"]
            params: list = [company_id]
            if zone_id is not None:
                conditions.append("zone_id = %s")
                params.append(zone_id)
            if person_code:
                conditions.append("person_code = %s")
                params.append(person_code)
            if active_only:
                conditions.append("active = 1")
            where = " AND ".join(conditions)
            return await db.fetch_all(
                f"""
                SELECT id, person_code, person_name, zone_id, site_id,
                       access_allowed, schedule_id,
                       valid_from, valid_until,
                       epi_profile_code, exposure_limit_min,
                       active, blocked, blocked_reason, blocked_at,
                       created_at, updated_at
                FROM visionapp_access_zone_persons
                WHERE {where}
                ORDER BY zone_id, person_name
                """,
                params,
            )
        except Exception as e:
            logger.error(f"[Repo] list_zone_persons: {e}")
            return []

    async def delete_zone_person(
        self,
        company_id: int,
        person_code: str,
        zone_id: int,
    ) -> bool:
        """Remove vínculo pessoa-zona."""
        try:
            await db.execute(
                """
                DELETE FROM visionapp_access_zone_persons
                WHERE company_id = %s AND person_code = %s AND zone_id = %s
                """,
                (company_id, person_code, zone_id),
            )
            return True
        except Exception as e:
            logger.error(f"[Repo] delete_zone_person: {e}")
            return False

    # =========================================================================
    # vision_video_jobs
    # =========================================================================

    async def create_video_job(
        self,
        company_id: int,
        source_type: str,               # file | youtube | rtsp_recording
        original_name: Optional[str] = None,
        source_url: Optional[str] = None,
        model_name: str = "best",
        confidence: float = 0.4,
        skip_frames: int = 5,
        detect_faces: bool = False,
    ) -> Optional[int]:
        """Cria um job de vídeo com status 'pending'. Retorna o id."""
        try:
            return await db.insert_get_id(
                """
                INSERT INTO vision_video_jobs (
                    company_id, source_type, source_url, original_name,
                    model_name, confidence, skip_frames, detect_faces,
                    status
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'pending')
                """,
                (
                    company_id, source_type, source_url, original_name,
                    model_name, confidence, skip_frames, detect_faces,
                ),
            )
        except Exception as e:
            logger.error(f"[Repo] create_video_job: {e}")
            return None

    async def start_video_job(self, job_id: int, frames_total: int = 0) -> bool:
        """Marca o job como 'processing' e registra total de frames."""
        try:
            await db.execute(
                """
                UPDATE vision_video_jobs SET
                    status       = 'processing',
                    frames_total = %s
                WHERE id = %s
                """,
                (frames_total, job_id),
            )
            return True
        except Exception as e:
            logger.error(f"[Repo] start_video_job: {e}")
            return False

    async def update_video_job_progress(
        self,
        job_id: int,
        frames_processed: int,
        frames_compliant: int,
    ) -> bool:
        """Atualiza progresso (frames processados / conformes) durante execução."""
        try:
            rate = round(frames_compliant / frames_processed, 4) if frames_processed else 0.0
            await db.execute(
                """
                UPDATE vision_video_jobs SET
                    frames_processed = %s,
                    frames_compliant = %s,
                    compliance_rate  = %s
                WHERE id = %s
                """,
                (frames_processed, frames_compliant, rate, job_id),
            )
            return True
        except Exception as e:
            logger.warning(f"[Repo] update_video_job_progress: {e}")
            return False

    async def complete_video_job(
        self,
        job_id: int,
        frames_total: int,
        frames_processed: int,
        frames_compliant: int,
        result_summary: dict,
        processing_ms: int,
    ) -> bool:
        """Fecha o job com status 'complete' e salva resultado final."""
        try:
            rate = round(frames_compliant / frames_processed, 4) if frames_processed else 0.0
            await db.execute(
                """
                UPDATE vision_video_jobs SET
                    status           = 'complete',
                    frames_total     = %s,
                    frames_processed = %s,
                    frames_compliant = %s,
                    compliance_rate  = %s,
                    result_summary   = %s,
                    processing_ms    = %s,
                    completed_at     = NOW()
                WHERE id = %s
                """,
                (
                    frames_total, frames_processed, frames_compliant,
                    rate, json.dumps(result_summary), processing_ms, job_id,
                ),
            )
            return True
        except Exception as e:
            logger.error(f"[Repo] complete_video_job: {e}")
            return False

    async def fail_video_job(
        self, job_id: int, error_message: str
    ) -> bool:
        """Fecha o job com status 'error'."""
        try:
            await db.execute(
                """
                UPDATE vision_video_jobs SET
                    status        = 'error',
                    error_message = %s,
                    completed_at  = NOW()
                WHERE id = %s
                """,
                (error_message, job_id),
            )
            return True
        except Exception as e:
            logger.error(f"[Repo] fail_video_job: {e}")
            return False

    async def get_video_job(
        self, company_id: int, job_id: int
    ) -> Optional[dict]:
        """Retorna um job pelo id."""
        try:
            return await db.fetch_one(
                """
                SELECT id, source_type, source_url, original_name,
                       model_name, confidence, skip_frames, detect_faces,
                       frames_total, frames_processed, frames_compliant,
                       compliance_rate, result_summary, status,
                       error_message, processing_ms, created_at, completed_at
                FROM vision_video_jobs
                WHERE id = %s AND company_id = %s
                """,
                (job_id, company_id),
            )
        except Exception as e:
            logger.error(f"[Repo] get_video_job: {e}")
            return None

    async def list_video_jobs(
        self,
        company_id: int,
        limit: int = 20,
        status: Optional[str] = None,  # pending | processing | complete | error
    ) -> list:
        """Lista jobs de vídeo com filtro opcional por status."""
        try:
            extra = "AND status = %s" if status else ""
            params = [company_id, status, limit] if status else [company_id, limit]
            return await db.fetch_all(
                f"""
                SELECT id, source_type, original_name, model_name,
                       frames_total, frames_processed, frames_compliant,
                       compliance_rate, status, error_message,
                       processing_ms, created_at, completed_at
                FROM vision_video_jobs
                WHERE company_id = %s {extra}
                ORDER BY created_at DESC LIMIT %s
                """,
                params,
            )
        except Exception as e:
            logger.error(f"[Repo] list_video_jobs: {e}")
            return []

    # =========================================================================
    # vision_validation_sessions  +  vision_validation_photos
    # =========================================================================

    async def create_validation_session(
        self,
        company_id: int,
        session_uuid: str,
        door_id: str,
        direction: str,                     # ENTRY | EXIT
        timeout_seconds: int = 30,
        camera_id: Optional[int] = None,
        zone_id: Optional[int] = None,
        compliance_mode: str = "majority",  # majority | all | best | worst
        photo_count_required: int = 3,
    ) -> Optional[int]:
        """
        Cria uma sessão de validação aberta.
        expires_at = NOW() + timeout_seconds.
        Retorna o id da sessão.
        """
        try:
            return await db.insert_get_id(
                """
                INSERT INTO vision_validation_sessions (
                    company_id, session_uuid, door_id, direction,
                    camera_id, zone_id,
                    timeout_seconds, expires_at,
                    compliance_mode, photo_count_required,
                    session_status, access_decision
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, DATE_ADD(NOW(), INTERVAL %s SECOND),
                    %s, %s,
                    'open', 'PENDING'
                )
                """,
                (
                    company_id, session_uuid, door_id, direction,
                    camera_id, zone_id,
                    timeout_seconds, timeout_seconds,
                    compliance_mode, photo_count_required,
                ),
            )
        except Exception as e:
            logger.error(f"[Repo] create_validation_session: {e}")
            return None

    async def get_validation_session(
        self, company_id: int, session_uuid: str
    ) -> Optional[dict]:
        """Retorna sessão pelo UUID."""
        try:
            return await db.fetch_one(
                """
                SELECT * FROM vision_validation_sessions
                WHERE company_id = %s AND session_uuid = %s
                """,
                (company_id, session_uuid),
            )
        except Exception as e:
            logger.error(f"[Repo] get_validation_session: {e}")
            return None

    async def get_open_session_by_door(
        self, company_id: int, door_id: str
    ) -> Optional[dict]:
        """Retorna sessão aberta mais recente para uma porta."""
        try:
            return await db.fetch_one(
                """
                SELECT * FROM vision_validation_sessions
                WHERE company_id = %s AND door_id = %s
                  AND session_status = 'open'
                  AND expires_at > NOW()
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (company_id, door_id),
            )
        except Exception as e:
            logger.error(f"[Repo] get_open_session_by_door: {e}")
            return None

    async def add_validation_photo(
        self,
        company_id: int,
        session_id: int,
        session_uuid: str,
        photo_seq: int,                     # 1, 2 ou 3
        filename: str,
        filepath: str,
        # face recognition
        face_detected: bool = False,
        face_confidence: Optional[float] = None,
        face_person_code: Optional[str] = None,
        face_bbox: Optional[dict] = None,   # {x, y, w, h}
        # EPI detection
        epi_compliant: Optional[bool] = None,
        compliance_score: Optional[float] = None,
        epi_units_required: int = 0,
        epi_units_detected: int = 0,
        epi_units_missing: int = 0,
        class_compliant: Optional[bool] = None,
        compliance_detail: Optional[dict] = None,
        raw_detections: Optional[list] = None,
        model_name: Optional[str] = None,
        processing_ms: Optional[int] = None,
        # imagem
        width: Optional[int] = None,
        height: Optional[int] = None,
        file_size_kb: Optional[int] = None,
        image_quality: Optional[float] = None,
    ) -> Optional[int]:
        """Insere uma foto da sessão de validação."""
        try:
            bbox = face_bbox or {}
            return await db.insert_get_id(
                """
                INSERT INTO vision_validation_photos (
                    company_id, session_id, session_uuid, photo_seq,
                    filename, filepath, file_size_kb, width, height, image_quality,
                    face_detected, face_confidence, face_person_code,
                    face_bbox_x, face_bbox_y, face_bbox_w, face_bbox_h,
                    epi_compliant, compliance_score,
                    epi_units_required, epi_units_detected, epi_units_missing,
                    class_compliant, compliance_detail, raw_detections,
                    model_name, processing_ms
                ) VALUES (
                    %s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,
                    %s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,
                    %s,%s,%s,
                    %s,%s,%s,
                    %s,%s
                )
                """,
                (
                    company_id, session_id, session_uuid, photo_seq,
                    filename, filepath, file_size_kb, width, height, image_quality,
                    face_detected, face_confidence, face_person_code,
                    bbox.get("x"), bbox.get("y"), bbox.get("w"), bbox.get("h"),
                    epi_compliant, compliance_score,
                    epi_units_required, epi_units_detected, epi_units_missing,
                    class_compliant,
                    json.dumps(compliance_detail) if compliance_detail else None,
                    json.dumps(raw_detections) if raw_detections else None,
                    model_name, processing_ms,
                ),
            )
        except Exception as e:
            logger.error(f"[Repo] add_validation_photo: {e}")
            return None

    async def increment_photo_count(
        self, session_id: int, company_id: int
    ) -> int:
        """Incrementa photo_count_received e retorna o novo valor."""
        try:
            await db.execute(
                """
                UPDATE vision_validation_sessions
                SET photo_count_received = photo_count_received + 1
                WHERE id = %s AND company_id = %s
                """,
                (session_id, company_id),
            )
            row = await db.fetch_one(
                "SELECT photo_count_received FROM vision_validation_sessions WHERE id = %s",
                (session_id,),
            )
            return (row or {}).get("photo_count_received", 0)
        except Exception as e:
            logger.error(f"[Repo] increment_photo_count: {e}")
            return 0

    async def close_validation_session(
        self,
        session_id: int,
        company_id: int,
        session_status: str,                # complete | timeout | error
        access_decision: str,               # GRANTED | DENIED_EPI | DENIED_FACE | ...
        epi_compliant: Optional[bool] = None,
        compliance_score: Optional[float] = None,
        face_confirmed: Optional[bool] = None,
        face_confidence_max: Optional[float] = None,
        person_code: Optional[str] = None,
        person_name: Optional[str] = None,
    ) -> bool:
        """Fecha a sessão com resultado consolidado."""
        try:
            await db.execute(
                """
                UPDATE vision_validation_sessions SET
                    session_status      = %s,
                    access_decision     = %s,
                    epi_compliant       = %s,
                    compliance_score    = %s,
                    face_confirmed      = %s,
                    face_confidence_max = %s,
                    person_code         = COALESCE(%s, person_code),
                    person_name         = COALESCE(%s, person_name),
                    closed_at           = NOW()
                WHERE id = %s AND company_id = %s
                """,
                (
                    session_status, access_decision,
                    epi_compliant, compliance_score,
                    face_confirmed, face_confidence_max,
                    person_code, person_name,
                    session_id, company_id,
                ),
            )
            return True
        except Exception as e:
            logger.error(f"[Repo] close_validation_session: {e}")
            return False

    async def expire_timed_out_sessions(self, company_id: int) -> int:
        """
        Fecha sessões abertas que passaram do expires_at.
        Retorna quantas foram fechadas.
        """
        try:
            await db.execute(
                """
                UPDATE vision_validation_sessions SET
                    session_status  = 'timeout',
                    access_decision = 'DENIED_FACE',
                    closed_at       = NOW()
                WHERE company_id = %s
                  AND session_status = 'open'
                  AND expires_at < NOW()
                """,
                (company_id,),
            )
            # Retorna rows affected (aproximado via SELECT)
            row = await db.fetch_one(
                """
                SELECT COUNT(*) AS cnt FROM vision_validation_sessions
                WHERE company_id = %s AND session_status = 'timeout'
                  AND closed_at >= DATE_SUB(NOW(), INTERVAL 5 SECOND)
                """,
                (company_id,),
            )
            return (row or {}).get("cnt", 0)
        except Exception as e:
            logger.error(f"[Repo] expire_timed_out_sessions: {e}")
            return 0

    # async def get_validation_photos(
    #     self, company_id: int, session_uuid: str
    # ) -> list:
    #     """Retorna todas as fotos de uma sessão."""
    #     try:
    #         return await db.fetch_all(
    #             """
    #             SELECT photo_seq, filename, filepath,
    #                    face_detected, face_confidence, face_person_code,
    #                    epi_compliant, compliance_score,
    #                    epi_units_required, epi_units_detected, epi_units_missing,
    #                    model_name, processing_ms, captured_at
    #             FROM vision_validation_photos
    #             WHERE company_id = %s AND session_uuid = %s
    #             ORDER BY photo_seq ASC
    #             """,
    #             (company_id, session_uuid),
    #         )
    #     except Exception as e:
    #         logger.error(f"[Repo] get_validation_photos: {e}")
    #         return []
    async def get_validation_photos(
        self, company_id: int, session_uuid: str
    ) -> list:
        """Retorna todas as fotos de uma sessão."""
        try:
            return await db.fetch_all(
                """
                SELECT
                    vvp.photo_seq,
                    vvp.filename,
                    vvp.filepath,
                    vvp.face_detected,
                    vvp.face_confidence,
                    vvp.face_person_code,
                    vp.person_name,
                    vvp.epi_compliant,
                    vvp.compliance_score,
                    vvp.epi_units_required,
                    vvp.epi_units_detected,
                    vvp.epi_units_missing,
                    vvp.model_name,
                    vvp.processing_ms,
                    vvp.captured_at
                FROM vision_validation_photos vvp
                LEFT JOIN vision_people vp
                    ON vp.person_code = vvp.face_person_code
                    AND vp.company_id = vvp.company_id
                WHERE
                    vvp.company_id = %s
                    AND vvp.session_uuid = %s
                ORDER BY vvp.photo_seq ASC
                """,
                (company_id, session_uuid),
            )
        except Exception as e:
            logger.error(f"[Repo] get_validation_photos: {e}")
            return []

# Singleton — importe nos outros módulos com:
#   from app.core.repository import repo
repo = VisionRepository()