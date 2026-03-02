# """
# SmartX Vision Platform v3 — MySQL Repository Layer
# ====================================================
# Mapeado diretamente do DDL real do banco ai_hub.

# Tabelas vision_*    → escritas pelo AI Hub (este sistema)
# Tabelas visionapp_* → escritas/lidas pelo SmartX Hub (Node-RED/Hub)

# Este arquivo cobre:
#   ESCRITA  → vision_detection_events, vision_epi_detections,
#              vision_alerts, vision_people, vision_face_photos,
#              vision_ppe_config, vision_training_runs, vision_models,
#              vision_stream_sessions, vision_snapshots, vision_datasets,
#              vision_compliance_hourly, vision_compliance_daily, vision_sync_log

#   LEITURA  → visionapp_ppe_config, visionapp_people,
#              visionapp_recognition_events (dados gerenciados pelo Hub)
# """

# import json
# from typing import Optional
# from loguru import logger

# from app.core.database import db


# class VisionRepository:

#     # =========================================================================
#     # vision_detection_events
#     # =========================================================================

#     async def save_detection(
#         self,
#         company_id: int,
#         result: dict,
#         camera_id: Optional[int] = None,
#         zone_id: Optional[int] = None,
#         snapshot_path: Optional[str] = None,
#         edge_device_id: Optional[str] = None,
#         model_name: Optional[str] = None,
#         confidence_threshold: float = 0.4,
#         source_type: str = "upload",
#     ) -> Optional[int]:
#         """
#         Insere em vision_detection_events.
#         Colunas reais: epi_required_count, epi_detected_count, epi_missing_count,
#                        compliance_score, missing_items (json), detections (json),
#                        faces (json), source_type (enum), sync_status, sync_priority
#         """
#         try:
#             detections = result.get("detections", [])
#             missing    = result.get("missing", [])
#             faces      = result.get("faces", [])
#             compliant  = bool(result.get("compliant", False))

#             epi_required = result.get("epi_required_count", len(detections))
#             epi_detected = result.get("epi_detected_count", len([d for d in detections if d.get("detected", True)]))
#             epi_missing  = len(missing)
#             score        = round(epi_detected / epi_required, 4) if epi_required else 1.0

#             # Prioridade de sync: 1=crítico não-conforme, 5=normal
#             sync_priority = 1 if not compliant else 5

#             # Melhor face reconhecida
#             person_code = person_name = None
#             if faces:
#                 best = max(faces, key=lambda f: f.get("confidence", 0))
#                 if best.get("recognized"):
#                     person_code = best.get("person_code")
#                     person_name = best.get("person_name")

#             event_id = await db.insert_get_id(
#                 """
#                 INSERT INTO vision_detection_events (
#                     company_id, compliant,
#                     epi_required_count, epi_detected_count, epi_missing_count,
#                     compliance_score, missing_items, detections, faces,
#                     snapshot_path, person_code, person_name,
#                     camera_id, zone_id, edge_device_id,
#                     model_name, confidence_threshold, processing_ms,
#                     source_type, sync_priority
#                 ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
#                 """,
#                 (
#                     company_id, compliant,
#                     epi_required, epi_detected, epi_missing,
#                     score,
#                     json.dumps(missing),
#                     json.dumps(detections),
#                     json.dumps(faces),
#                     snapshot_path, person_code, person_name,
#                     camera_id, zone_id, edge_device_id,
#                     model_name or result.get("model_name"),
#                     confidence_threshold,
#                     result.get("processing_ms", 0),
#                     source_type, sync_priority,
#                 ),
#             )

#             if event_id and detections:
#                 await self._save_epi_detections(company_id, event_id, detections, missing)

#             return event_id

#         except Exception as e:
#             logger.error(f"[Repo] save_detection: {e}")
#             return None

#     async def _save_epi_detections(
#         self, company_id: int, event_id: int, detections: list, missing: list
#     ):
#         """
#         Agrega por class_name → insere em vision_epi_detections.
#         Colunas reais: quantity_required, quantity_detected, quantity_missing,
#                        class_compliant, best_confidence, avg_confidence, all_instances (json)
#         """
#         try:
#             classes: dict = {}
#             for det in detections:
#                 cn = det.get("class_name", "unknown")
#                 if cn not in classes:
#                     classes[cn] = {
#                         "instances": [],
#                         "required": det.get("quantity_required", 1),
#                     }
#                 classes[cn]["instances"].append(det)

#             for class_name, data in classes.items():
#                 instances  = data["instances"]
#                 qty_req    = data["required"]
#                 qty_det    = len(instances)
#                 qty_miss   = max(0, qty_req - qty_det)
#                 compliant  = qty_det >= qty_req
#                 confs      = [i.get("confidence", 0) for i in instances if i.get("confidence")]
#                 best_conf  = max(confs) if confs else None
#                 avg_conf   = round(sum(confs) / len(confs), 4) if confs else None

#                 await db.execute(
#                     """
#                     INSERT INTO vision_epi_detections (
#                         company_id, event_id, class_name,
#                         quantity_required, quantity_detected, quantity_missing,
#                         class_compliant, best_confidence, avg_confidence, all_instances
#                     ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
#                     """,
#                     (
#                         company_id, event_id, class_name,
#                         qty_req, qty_det, qty_miss,
#                         compliant, best_conf, avg_conf,
#                         json.dumps(instances),
#                     ),
#                 )
#         except Exception as e:
#             logger.warning(f"[Repo] _save_epi_detections (não crítico): {e}")

#     async def get_recent_detections(
#         self, company_id: int, limit: int = 50, only_noncompliant: bool = False
#     ) -> list:
#         try:
#             extra = "AND compliant = 0" if only_noncompliant else ""
#             return await db.fetch_all(
#                 f"""
#                 SELECT id, compliant, epi_required_count, epi_detected_count,
#                        epi_missing_count, compliance_score, missing_items,
#                        detections, faces, snapshot_path, person_code, person_name,
#                        camera_id, zone_id, model_name, processing_ms,
#                        source_type, sync_status, created_at
#                 FROM vision_detection_events
#                 WHERE company_id = %s {extra}
#                 ORDER BY created_at DESC LIMIT %s
#                 """,
#                 (company_id, limit),
#             )
#         except Exception as e:
#             logger.error(f"[Repo] get_recent_detections: {e}")
#             return []

#     # =========================================================================
#     # vision_alerts
#     # =========================================================================

#     async def save_alert(
#         self,
#         company_id: int,
#         alert_type: str,
#         details: dict,
#         severity: str = "medium",       # low | medium | high | critical
#         person_code: Optional[str] = None,
#         camera_id: Optional[int] = None,
#         zone_id: Optional[int] = None,
#     ) -> Optional[int]:
#         try:
#             return await db.insert_get_id(
#                 """
#                 INSERT INTO vision_alerts
#                     (company_id, alert_type, severity, details,
#                      person_code, camera_id, zone_id)
#                 VALUES (%s,%s,%s,%s,%s,%s,%s)
#                 """,
#                 (company_id, alert_type, severity, json.dumps(details),
#                  person_code, camera_id, zone_id),
#             )
#         except Exception as e:
#             logger.error(f"[Repo] save_alert: {e}")
#             return None

#     async def get_alerts(
#         self, company_id: int, limit: int = 50, unresolved_only: bool = False
#     ) -> list:
#         try:
#             extra = "AND resolved = 0" if unresolved_only else ""
#             return await db.fetch_all(
#                 f"""
#                 SELECT id, alert_type, severity, details, person_code,
#                        camera_id, zone_id, acknowledged, resolved,
#                        sync_status, created_at
#                 FROM vision_alerts
#                 WHERE company_id = %s {extra}
#                 ORDER BY created_at DESC LIMIT %s
#                 """,
#                 (company_id, limit),
#             )
#         except Exception as e:
#             logger.error(f"[Repo] get_alerts: {e}")
#             return []

#     # =========================================================================
#     # vision_people  +  vision_face_photos
#     # =========================================================================

#     async def upsert_person(
#         self,
#         company_id: int,
#         person_code: str,
#         person_name: str,
#         badge_id: str = "",
#         department: str = "",
#     ) -> Optional[int]:
#         """
#         Colunas reais da vision_people:
#         person_code, person_name, badge_id, department, face_photos_count,
#         face_embedding_path, photo_count_for_validation, compliance_mode,
#         face_recognition_on_exit, last_entry_at, last_exit_at, is_inside,
#         current_session_id, active, sync_status, sync_direction
#         """
#         try:
#             return await db.insert_get_id(
#                 """
#                 INSERT INTO vision_people
#                     (company_id, person_code, person_name, badge_id, department)
#                 VALUES (%s,%s,%s,%s,%s)
#                 ON DUPLICATE KEY UPDATE
#                     person_name  = VALUES(person_name),
#                     badge_id     = VALUES(badge_id),
#                     department   = VALUES(department),
#                     updated_at   = CURRENT_TIMESTAMP
#                 """,
#                 (company_id, person_code, person_name, badge_id, department),
#             )
#         except Exception as e:
#             logger.error(f"[Repo] upsert_person: {e}")
#             return None

#     async def save_face_photo(
#         self,
#         company_id: int,
#         person_code: str,
#         filename: str,
#         filepath: str,
#         quality_score: Optional[float] = None,
#         embedding: Optional[list] = None,
#     ) -> Optional[int]:
#         """
#         Colunas reais da vision_face_photos:
#         filename, filepath, embedding (json), quality_score,
#         registered_at, sync_status, hub_photo_id
#         NÃO tem photo_index (era do schema antigo).
#         """
#         try:
#             photo_id = await db.insert_get_id(
#                 """
#                 INSERT INTO vision_face_photos
#                     (company_id, person_code, filename, filepath,
#                      quality_score, embedding)
#                 VALUES (%s,%s,%s,%s,%s,%s)
#                 """,
#                 (
#                     company_id, person_code, filename, filepath,
#                     quality_score,
#                     json.dumps(embedding) if embedding else None,
#                 ),
#             )
#             # Incrementa contador na vision_people
#             if photo_id:
#                 await db.execute(
#                     """
#                     UPDATE vision_people
#                     SET face_photos_count = face_photos_count + 1,
#                         updated_at = CURRENT_TIMESTAMP
#                     WHERE company_id = %s AND person_code = %s
#                     """,
#                     (company_id, person_code),
#                 )
#             return photo_id
#         except Exception as e:
#             logger.error(f"[Repo] save_face_photo: {e}")
#             return None

#     async def list_people(self, company_id: int, active_only: bool = True) -> list:
#         try:
#             extra = "AND active = 1" if active_only else ""
#             return await db.fetch_all(
#                 f"""
#                 SELECT person_code, person_name, badge_id, department,
#                        face_photos_count, active,
#                        last_entry_at, last_exit_at, is_inside, created_at
#                 FROM vision_people
#                 WHERE company_id = %s {extra}
#                 ORDER BY person_name
#                 """,
#                 (company_id,),
#             )
#         except Exception as e:
#             logger.error(f"[Repo] list_people: {e}")
#             return []

#     async def get_person(self, company_id: int, person_code: str) -> Optional[dict]:
#         try:
#             return await db.fetch_one(
#                 """
#                 SELECT person_code, person_name, badge_id, department,
#                        face_photos_count, active, is_inside,
#                        last_entry_at, last_exit_at,
#                        compliance_mode, photo_count_for_validation
#                 FROM vision_people
#                 WHERE company_id = %s AND person_code = %s
#                 """,
#                 (company_id, person_code),
#             )
#         except Exception as e:
#             logger.error(f"[Repo] get_person: {e}")
#             return None

#     async def get_face_photos(self, company_id: int, person_code: str) -> list:
#         try:
#             return await db.fetch_all(
#                 """
#                 SELECT id, filename, filepath, quality_score, registered_at
#                 FROM vision_face_photos
#                 WHERE company_id = %s AND person_code = %s
#                 ORDER BY registered_at ASC
#                 """,
#                 (company_id, person_code),
#             )
#         except Exception as e:
#             logger.error(f"[Repo] get_face_photos: {e}")
#             return []

#     # =========================================================================
#     # vision_ppe_config  (escrita) + fallback visionapp_ppe_config (leitura)
#     # =========================================================================

#     async def save_ppe_config(self, company_id: int, config: dict) -> bool:
#         """
#         config = {
#           "helmet": {"enabled": True, "required": True, "min_quantity": 1,
#                      "confidence_min": 0.4, "display_name": "Capacete"},
#           ...
#         }
#         Colunas reais: class_name, display_name, body_region, enabled,
#                        required, min_quantity, max_quantity, confidence_min, compliance_note
#         """
#         try:
#             for class_name, cfg in config.items():
#                 if isinstance(cfg, bool):
#                     cfg = {"enabled": cfg}
#                 await db.execute(
#                     """
#                     INSERT INTO vision_ppe_config
#                         (company_id, class_name, enabled, required,
#                          min_quantity, confidence_min, display_name)
#                     VALUES (%s,%s,%s,%s,%s,%s,%s)
#                     ON DUPLICATE KEY UPDATE
#                         enabled        = VALUES(enabled),
#                         required       = VALUES(required),
#                         min_quantity   = VALUES(min_quantity),
#                         confidence_min = VALUES(confidence_min),
#                         display_name   = VALUES(display_name),
#                         updated_at     = CURRENT_TIMESTAMP
#                     """,
#                     (
#                         company_id, class_name,
#                         cfg.get("enabled", True),
#                         cfg.get("required", True),
#                         cfg.get("min_quantity", 1),
#                         cfg.get("confidence_min", 0.4),
#                         cfg.get("display_name"),
#                     ),
#                 )
#             return True
#         except Exception as e:
#             logger.error(f"[Repo] save_ppe_config: {e}")
#             return False

#     async def get_ppe_config(self, company_id: int) -> Optional[dict]:
#         """
#         1º tenta vision_ppe_config (AI Hub local)
#         2º fallback para visionapp_ppe_config (Hub — fonte de verdade)
#         3º retorna None → usa config de arquivo
#         """
#         try:
#             for table in ("vision_ppe_config", "visionapp_ppe_config"):
#                 rows = await db.fetch_all(
#                     f"""
#                     SELECT class_name, enabled, required, min_quantity,
#                            max_quantity, confidence_min, display_name, body_region
#                     FROM {table}
#                     WHERE company_id = %s
#                     """,
#                     (company_id,),
#                 )
#                 if rows:
#                     return {
#                         r["class_name"]: {
#                             "enabled":        bool(r["enabled"]),
#                             "required":       bool(r["required"]),
#                             "min_quantity":   r["min_quantity"],
#                             "max_quantity":   r.get("max_quantity"),
#                             "confidence_min": r["confidence_min"],
#                             "display_name":   r.get("display_name"),
#                             "body_region":    r.get("body_region"),
#                         }
#                         for r in rows
#                     }
#             return None
#         except Exception as e:
#             logger.error(f"[Repo] get_ppe_config: {e}")
#             return None

#     # =========================================================================
#     # vision_training_runs
#     # =========================================================================

#     async def create_training_run(
#         self,
#         company_id: int,
#         base_model: str,
#         epochs: int,
#         batch_size: int,
#         img_size: int,
#         classes: dict,
#         dataset_id: Optional[int] = None,
#     ) -> Optional[int]:
#         """
#         Colunas reais: dataset_id, base_model, epochs, batch_size,
#                        img_size, classes, status, started_at
#         """
#         try:
#             return await db.insert_get_id(
#                 """
#                 INSERT INTO vision_training_runs
#                     (company_id, dataset_id, base_model, epochs,
#                      batch_size, img_size, classes, status, started_at)
#                 VALUES (%s,%s,%s,%s,%s,%s,%s,'training', NOW())
#                 """,
#                 (company_id, dataset_id, base_model, epochs,
#                  batch_size, img_size, json.dumps(classes)),
#             )
#         except Exception as e:
#             logger.error(f"[Repo] create_training_run: {e}")
#             return None

#     async def update_training_run(
#         self,
#         run_id: int,
#         status: str,                        # pending | training | complete | error
#         best_map50: Optional[float] = None,
#         best_map50_95: Optional[float] = None,
#         model_path: Optional[str] = None,
#         error_message: Optional[str] = None,
#     ) -> bool:
#         try:
#             set_completed = "completed_at = NOW()," if status in ("complete", "error") else ""
#             await db.execute(
#                 f"""
#                 UPDATE vision_training_runs SET
#                     status        = %s,
#                     best_map50    = COALESCE(%s, best_map50),
#                     best_map50_95 = COALESCE(%s, best_map50_95),
#                     model_path    = COALESCE(%s, model_path),
#                     error_message = COALESCE(%s, error_message)
#                     {(',' + set_completed).rstrip(',')}
#                 WHERE id = %s
#                 """,
#                 (status, best_map50, best_map50_95, model_path, error_message, run_id),
#             )
#             return True
#         except Exception as e:
#             logger.error(f"[Repo] update_training_run: {e}")
#             return False

#     async def get_training_history(self, company_id: int, limit: int = 20) -> list:
#         try:
#             return await db.fetch_all(
#                 """
#                 SELECT id, dataset_id, base_model, epochs, batch_size,
#                        img_size, classes, status,
#                        best_map50, best_map50_95, model_path,
#                        error_message, started_at, completed_at, created_at
#                 FROM vision_training_runs
#                 WHERE company_id = %s
#                 ORDER BY created_at DESC LIMIT %s
#                 """,
#                 (company_id, limit),
#             )
#         except Exception as e:
#             logger.error(f"[Repo] get_training_history: {e}")
#             return []

#     # =========================================================================
#     # vision_models
#     # =========================================================================

#     async def save_model(
#         self,
#         company_id: int,
#         model_name: str,
#         filename: str,
#         filepath: str,
#         file_size_mb: float = 0,
#         base_model: Optional[str] = None,
#         training_run: Optional[int] = None,
#         map50: Optional[float] = None,
#         map50_95: Optional[float] = None,
#         classes: Optional[dict] = None,
#     ) -> Optional[int]:
#         """
#         Colunas reais: model_name, filename, filepath, file_size_mb,
#                        base_model, classes, map50, map50_95, training_run, active,
#                        sync_status, sync_direction
#         """
#         try:
#             return await db.insert_get_id(
#                 """
#                 INSERT INTO vision_models
#                     (company_id, model_name, filename, filepath,
#                      file_size_mb, base_model, training_run,
#                      map50, map50_95, classes)
#                 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
#                 ON DUPLICATE KEY UPDATE
#                     filename     = VALUES(filename),
#                     filepath     = VALUES(filepath),
#                     file_size_mb = VALUES(file_size_mb),
#                     map50        = VALUES(map50),
#                     map50_95     = VALUES(map50_95)
#                 """,
#                 (
#                     company_id, model_name, filename, filepath,
#                     file_size_mb, base_model, training_run,
#                     map50, map50_95, json.dumps(classes or {}),
#                 ),
#             )
#         except Exception as e:
#             logger.error(f"[Repo] save_model: {e}")
#             return None

#     async def list_models(self, company_id: int, active_only: bool = True) -> list:
#         try:
#             extra = "AND active = 1" if active_only else ""
#             return await db.fetch_all(
#                 f"""
#                 SELECT id, model_name, filename, filepath, file_size_mb,
#                        base_model, classes, map50, map50_95,
#                        training_run, active, sync_status, created_at
#                 FROM vision_models
#                 WHERE company_id = %s {extra}
#                 ORDER BY created_at DESC
#                 """,
#                 (company_id,),
#             )
#         except Exception as e:
#             logger.error(f"[Repo] list_models: {e}")
#             return []

#     # =========================================================================
#     # vision_stream_sessions
#     # =========================================================================

#     async def save_stream_session(
#         self,
#         company_id: int,
#         session_id: str,
#         source_url: str,
#         source_type: str,               # rtsp | youtube | webcam | file | browser_camera
#         model_name: str = "best",
#         confidence: float = 0.4,
#         detect_faces: bool = False,
#     ) -> bool:
#         """
#         Colunas reais: session_id (UNIQUE), source_url, source_type,
#                        model_name, confidence, detect_faces, frame_count,
#                        avg_fps, compliant_pct, status, started_at
#         """
#         try:
#             await db.execute(
#                 """
#                 INSERT IGNORE INTO vision_stream_sessions
#                     (company_id, session_id, source_url, source_type,
#                      model_name, confidence, detect_faces, status)
#                 VALUES (%s,%s,%s,%s,%s,%s,%s,'active')
#                 """,
#                 (company_id, session_id, source_url, source_type,
#                  model_name, confidence, detect_faces),
#             )
#             return True
#         except Exception as e:
#             logger.warning(f"[Repo] save_stream_session: {e}")
#             return False

#     async def close_stream_session(
#         self,
#         session_id: str,
#         frame_count: int = 0,
#         avg_fps: float = 0,
#         compliant_pct: Optional[float] = None,
#         status: str = "stopped",
#         error_message: Optional[str] = None,
#     ) -> bool:
#         try:
#             await db.execute(
#                 """
#                 UPDATE vision_stream_sessions SET
#                     frame_count   = %s,
#                     avg_fps       = %s,
#                     compliant_pct = COALESCE(%s, compliant_pct),
#                     status        = %s,
#                     error_message = COALESCE(%s, error_message),
#                     stopped_at    = NOW()
#                 WHERE session_id = %s
#                 """,
#                 (frame_count, avg_fps, compliant_pct, status, error_message, session_id),
#             )
#             return True
#         except Exception as e:
#             logger.warning(f"[Repo] close_stream_session: {e}")
#             return False

#     # =========================================================================
#     # vision_snapshots
#     # =========================================================================

#     async def save_snapshot(
#         self,
#         company_id: int,
#         filename: str,
#         filepath: str,
#         snapshot_type: str = "EPI_DETECTION",
#         source_type: str = "stream",
#         event_id: Optional[int] = None,
#         session_id: Optional[int] = None,
#         file_size_kb: Optional[int] = None,
#         width: Optional[int] = None,
#         height: Optional[int] = None,
#     ) -> Optional[int]:
#         """
#         snapshot_type: EPI_DETECTION | FACE | ALERT | TRAINING | DOOR_OPEN | DOOR_CLOSE
#         source_type:   upload | browser_camera | stream | video
#         """
#         try:
#             return await db.insert_get_id(
#                 """
#                 INSERT INTO vision_snapshots
#                     (company_id, event_id, session_id, filename, filepath,
#                      file_size_kb, width, height, snapshot_type, source_type)
#                 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
#                 """,
#                 (company_id, event_id, session_id, filename, filepath,
#                  file_size_kb, width, height, snapshot_type, source_type),
#             )
#         except Exception as e:
#             logger.warning(f"[Repo] save_snapshot: {e}")
#             return None

#     # =========================================================================
#     # vision_datasets
#     # =========================================================================

#     async def save_dataset(
#         self,
#         company_id: int,
#         train_count: int,
#         valid_count: int,
#         classes: dict,
#         yaml_path: str,
#         train_split: float = 0.8,
#         valid_split: float = 0.15,
#     ) -> Optional[int]:
#         try:
#             return await db.insert_get_id(
#                 """
#                 INSERT INTO vision_datasets
#                     (company_id, train_count, valid_count, total_pairs,
#                      classes, train_split, valid_split, yaml_path)
#                 VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
#                 """,
#                 (
#                     company_id, train_count, valid_count,
#                     train_count + valid_count,
#                     json.dumps(classes),
#                     train_split, valid_split, yaml_path,
#                 ),
#             )
#         except Exception as e:
#             logger.error(f"[Repo] save_dataset: {e}")
#             return None

#     # =========================================================================
#     # vision_compliance_hourly / vision_compliance_daily (upsert agregado)
#     # =========================================================================

#     async def upsert_compliance_hourly(
#         self,
#         company_id: int,
#         hour_ts: str,           # "2025-03-01 14:00:00"
#         total: int,
#         compliant: int,
#         zone_id: Optional[int] = None,
#     ) -> bool:
#         """uq_hour_zone: (company_id, hour_ts, zone_id)"""
#         try:
#             rate = round(compliant / total, 4) if total else 0.0
#             await db.execute(
#                 """
#                 INSERT INTO vision_compliance_hourly
#                     (company_id, hour_ts, zone_id,
#                      total_sessions, compliant_count, compliance_rate)
#                 VALUES (%s,%s,%s,%s,%s,%s)
#                 ON DUPLICATE KEY UPDATE
#                     total_sessions  = total_sessions  + VALUES(total_sessions),
#                     compliant_count = compliant_count + VALUES(compliant_count),
#                     compliance_rate = compliant_count / total_sessions
#                 """,
#                 (company_id, hour_ts, zone_id, total, compliant, rate),
#             )
#             return True
#         except Exception as e:
#             logger.warning(f"[Repo] upsert_compliance_hourly: {e}")
#             return False

#     async def upsert_compliance_daily(
#         self,
#         company_id: int,
#         date: str,              # "2025-03-01"
#         total: int,
#         compliant: int,
#         zone_id: Optional[int] = None,
#     ) -> bool:
#         """uq_day_zone: (company_id, date, zone_id)"""
#         try:
#             rate = round(compliant / total, 4) if total else 0.0
#             await db.execute(
#                 """
#                 INSERT INTO vision_compliance_daily
#                     (company_id, date, zone_id,
#                      total_sessions, compliant_count, compliance_rate)
#                 VALUES (%s,%s,%s,%s,%s,%s)
#                 ON DUPLICATE KEY UPDATE
#                     total_sessions  = total_sessions  + VALUES(total_sessions),
#                     compliant_count = compliant_count + VALUES(compliant_count),
#                     compliance_rate = compliant_count / total_sessions
#                 """,
#                 (company_id, date, zone_id, total, compliant, rate),
#             )
#             return True
#         except Exception as e:
#             logger.warning(f"[Repo] upsert_compliance_daily: {e}")
#             return False

#     # =========================================================================
#     # vision_sync_log
#     # =========================================================================

#     async def log_sync(
#         self,
#         entity_type: str,
#         entity_id: int,
#         direction: str,         # edge_to_hub | hub_to_edge
#         status: str,            # success | error | partial | skipped
#         http_status: Optional[int] = None,
#         duration_ms: Optional[int] = None,
#         error_message: Optional[str] = None,
#         hub_entity_id: Optional[int] = None,
#         batch_id: Optional[str] = None,
#     ) -> bool:
#         try:
#             await db.execute(
#                 """
#                 INSERT INTO vision_sync_log
#                     (sync_direction, entity_type, entity_id, hub_entity_id,
#                      status, http_status, duration_ms, error_message, batch_id)
#                 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
#                 """,
#                 (direction, entity_type, entity_id, hub_entity_id,
#                  status, http_status, duration_ms, error_message, batch_id),
#             )
#             return True
#         except Exception as e:
#             logger.warning(f"[Repo] log_sync: {e}")
#             return False

#     # =========================================================================
#     # LEITURA visionapp_* (gerenciados pelo Hub)
#     # =========================================================================

#     async def get_hub_people(self, company_id: int, active_only: bool = True) -> list:
#         """visionapp_people — fonte de verdade do Hub."""
#         try:
#             extra = "AND active = 1" if active_only else ""
#             return await db.fetch_all(
#                 f"""
#                 SELECT person_code, person_name, badge_id, department,
#                        face_photos_count, is_inside, last_entry_at, last_exit_at
#                 FROM visionapp_people
#                 WHERE company_id = %s {extra}
#                 ORDER BY person_name
#                 """,
#                 (company_id,),
#             )
#         except Exception as e:
#             logger.error(f"[Repo] get_hub_people: {e}")
#             return []

#     async def get_hub_recognition_events(
#         self, company_id: int, limit: int = 50
#     ) -> list:
#         """visionapp_recognition_events — eventos consolidados do Hub."""
#         try:
#             return await db.fetch_all(
#                 """
#                 SELECT id, event_timestamp, camera_id, zone_id,
#                        person_code, person_name,
#                        face_recognized, face_confidence,
#                        epi_compliant, compliance_score, compliance_detail,
#                        epi_units_required, epi_units_detected, epi_units_missing,
#                        access_decision, processing_time_ms, created_at
#                 FROM visionapp_recognition_events
#                 WHERE company_id = %s
#                 ORDER BY event_timestamp DESC LIMIT %s
#                 """,
#                 (company_id, limit),
#             )
#         except Exception as e:
#             logger.error(f"[Repo] get_hub_recognition_events: {e}")
#             return []

#     # =========================================================================
#     # ANALYTICS / DASHBOARD
#     # =========================================================================

#     async def get_dashboard_stats(self, company_id: int) -> dict:
#         try:
#             today = await db.fetch_one(
#                 """
#                 SELECT COUNT(*) AS total,
#                        SUM(compliant) AS compliant,
#                        ROUND(AVG(compliant)*100,1) AS rate
#                 FROM vision_detection_events
#                 WHERE company_id = %s AND DATE(created_at) = CURDATE()
#                 """,
#                 (company_id,),
#             )
#             week = await db.fetch_one(
#                 """
#                 SELECT COUNT(*) AS total,
#                        ROUND(AVG(compliant)*100,1) AS rate
#                 FROM vision_detection_events
#                 WHERE company_id = %s
#                   AND created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
#                 """,
#                 (company_id,),
#             )
#             people = await db.fetch_one(
#                 "SELECT COUNT(*) AS cnt FROM vision_people WHERE company_id=%s AND active=1",
#                 (company_id,),
#             )
#             alerts = await db.fetch_one(
#                 "SELECT COUNT(*) AS cnt FROM vision_alerts WHERE company_id=%s AND resolved=0",
#                 (company_id,),
#             )
#             models = await db.fetch_one(
#                 "SELECT COUNT(*) AS cnt FROM vision_models WHERE company_id=%s AND active=1",
#                 (company_id,),
#             )
#             return {
#                 "today":        today or {},
#                 "week":         week or {},
#                 "people_count": (people or {}).get("cnt", 0),
#                 "alerts_open":  (alerts or {}).get("cnt", 0),
#                 "models_count": (models or {}).get("cnt", 0),
#             }
#         except Exception as e:
#             logger.error(f"[Repo] get_dashboard_stats: {e}")
#             return {"today": {}, "week": {}, "people_count": 0, "alerts_open": 0, "models_count": 0}

#     async def get_hourly_compliance(self, company_id: int, hours: int = 24) -> list:
#         """
#         Usa vision_compliance_hourly (pré-agregado) se disponível.
#         Fallback: agrega em tempo real de vision_detection_events.
#         """
#         try:
#             rows = await db.fetch_all(
#                 """
#                 SELECT hour_ts, total_sessions, compliant_count, compliance_rate
#                 FROM vision_compliance_hourly
#                 WHERE company_id = %s
#                   AND hour_ts >= DATE_SUB(NOW(), INTERVAL %s HOUR)
#                 ORDER BY hour_ts ASC
#                 """,
#                 (company_id, hours),
#             )
#             if rows:
#                 return rows

#             return await db.fetch_all(
#                 """
#                 SELECT
#                     DATE_FORMAT(created_at,'%Y-%m-%d %H:00:00') AS hour_ts,
#                     COUNT(*)       AS total_sessions,
#                     SUM(compliant) AS compliant_count,
#                     ROUND(AVG(compliant)*100,1) AS compliance_rate
#                 FROM vision_detection_events
#                 WHERE company_id = %s
#                   AND created_at >= DATE_SUB(NOW(), INTERVAL %s HOUR)
#                 GROUP BY hour_ts ORDER BY hour_ts ASC
#                 """,
#                 (company_id, hours),
#             )
#         except Exception as e:
#             logger.error(f"[Repo] get_hourly_compliance: {e}")
#             return []

#     async def get_missing_ppe_ranking(self, company_id: int, days: int = 7) -> list:
#         """
#         Usa vision_epi_detections (eficiente).
#         Fallback: parseia JSON de vision_detection_events.
#         """
#         try:
#             rows = await db.fetch_all(
#                 """
#                 SELECT class_name, SUM(quantity_missing) AS total_missing
#                 FROM vision_epi_detections
#                 WHERE company_id = %s
#                   AND quantity_missing > 0
#                   AND created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
#                 GROUP BY class_name
#                 ORDER BY total_missing DESC
#                 """,
#                 (company_id, days),
#             )
#             if rows:
#                 return rows

#             raw = await db.fetch_all(
#                 """
#                 SELECT missing_items FROM vision_detection_events
#                 WHERE company_id=%s AND compliant=0
#                   AND created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
#                 """,
#                 (company_id, days),
#             )
#             counts: dict = {}
#             for r in raw:
#                 for item in json.loads(r.get("missing_items") or "[]"):
#                     counts[item] = counts.get(item, 0) + 1
#             return sorted(
#                 [{"class_name": k, "total_missing": v} for k, v in counts.items()],
#                 key=lambda x: x["total_missing"], reverse=True,
#             )
#         except Exception as e:
#             logger.error(f"[Repo] get_missing_ppe_ranking: {e}")
#             return []

#     async def get_compliance_summary(self, company_id: int, days: int = 7) -> dict:
#         try:
#             return await db.fetch_one(
#                 """
#                 SELECT
#                     COUNT(*)                    AS total,
#                     SUM(compliant)              AS compliant_count,
#                     SUM(NOT compliant)          AS noncompliant_count,
#                     ROUND(AVG(compliant)*100,1) AS compliance_rate,
#                     AVG(compliance_score)       AS avg_score,
#                     AVG(processing_ms)          AS avg_processing_ms
#                 FROM vision_detection_events
#                 WHERE company_id = %s
#                   AND created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
#                 """,
#                 (company_id, days),
#             ) or {}
#         except Exception as e:
#             logger.error(f"[Repo] get_compliance_summary: {e}")
#             return {}


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
             vision_annotations

  LEITURA  → visionapp_ppe_config, visionapp_people,
             visionapp_recognition_events (dados gerenciados pelo Hub)
"""

import json
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
        """
        Insere em vision_detection_events.
        Colunas reais: epi_required_count, epi_detected_count, epi_missing_count,
                       compliance_score, missing_items (json), detections (json),
                       faces (json), source_type (enum), sync_status, sync_priority
        """
        try:
            detections = result.get("detections", [])
            missing    = result.get("missing", [])
            faces      = result.get("faces", [])
            compliant  = bool(result.get("compliant", False))

            epi_required = result.get("epi_required_count", len(detections))
            epi_detected = result.get("epi_detected_count", len([d for d in detections if d.get("detected", True)]))
            epi_missing  = len(missing)
            score        = round(epi_detected / epi_required, 4) if epi_required else 1.0

            # Prioridade de sync: 1=crítico não-conforme, 5=normal
            sync_priority = 1 if not compliant else 5

            # Melhor face reconhecida
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
        """
        Agrega por class_name → insere em vision_epi_detections.
        Colunas reais: quantity_required, quantity_detected, quantity_missing,
                       class_compliant, best_confidence, avg_confidence, all_instances (json)
        """
        try:
            classes: dict = {}
            for det in detections:
                cn = det.get("class_name", "unknown")
                if cn not in classes:
                    classes[cn] = {
                        "instances": [],
                        "required": det.get("quantity_required", 1),
                    }
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
        severity: str = "medium",       # low | medium | high | critical
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

    async def upsert_person(
        self,
        company_id: int,
        person_code: str,
        person_name: str,
        badge_id: str = "",
        department: str = "",
    ) -> Optional[int]:
        """
        Colunas reais da vision_people:
        person_code, person_name, badge_id, department, face_photos_count,
        face_embedding_path, photo_count_for_validation, compliance_mode,
        face_recognition_on_exit, last_entry_at, last_exit_at, is_inside,
        current_session_id, active, sync_status, sync_direction
        """
        try:
            return await db.insert_get_id(
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
        """
        Colunas reais da vision_face_photos:
        filename, filepath, embedding (json), quality_score,
        registered_at, sync_status, hub_photo_id
        NÃO tem photo_index (era do schema antigo).
        """
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
            # Incrementa contador na vision_people
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
    # vision_ppe_config  (escrita) + fallback visionapp_ppe_config (leitura)
    # =========================================================================

    async def save_ppe_config(self, company_id: int, config: dict) -> bool:
        """
        config = {
          "helmet": {"enabled": True, "required": True, "min_quantity": 1,
                     "confidence_min": 0.4, "display_name": "Capacete"},
          ...
        }
        Colunas reais: class_name, display_name, body_region, enabled,
                       required, min_quantity, max_quantity, confidence_min, compliance_note
        """
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
        """
        1º tenta vision_ppe_config (AI Hub local)
        2º fallback para visionapp_ppe_config (Hub — fonte de verdade)
        3º retorna None → usa config de arquivo
        """
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
        """
        Colunas reais: dataset_id, base_model, epochs, batch_size,
                       img_size, classes, status, started_at
        """
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
        status: str,                        # pending | training | complete | error
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
    # vision_models
    # =========================================================================

    async def save_model(
        self,
        company_id: int,
        model_name: str,
        filename: str,
        filepath: str,
        file_size_mb: float = 0,
        base_model: Optional[str] = None,
        training_run: Optional[int] = None,
        map50: Optional[float] = None,
        map50_95: Optional[float] = None,
        classes: Optional[dict] = None,
    ) -> Optional[int]:
        """
        Colunas reais: model_name, filename, filepath, file_size_mb,
                       base_model, classes, map50, map50_95, training_run, active,
                       sync_status, sync_direction
        """
        try:
            return await db.insert_get_id(
                """
                INSERT INTO vision_models
                    (company_id, model_name, filename, filepath,
                     file_size_mb, base_model, training_run,
                     map50, map50_95, classes)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                    filename     = VALUES(filename),
                    filepath     = VALUES(filepath),
                    file_size_mb = VALUES(file_size_mb),
                    map50        = VALUES(map50),
                    map50_95     = VALUES(map50_95)
                """,
                (
                    company_id, model_name, filename, filepath,
                    file_size_mb, base_model, training_run,
                    map50, map50_95, json.dumps(classes or {}),
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
        source_type: str,               # rtsp | youtube | webcam | file | browser_camera
        model_name: str = "best",
        confidence: float = 0.4,
        detect_faces: bool = False,
    ) -> bool:
        """
        Colunas reais: session_id (UNIQUE), source_url, source_type,
                       model_name, confidence, detect_faces, frame_count,
                       avg_fps, compliant_pct, status, started_at
        """
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
        """
        snapshot_type: EPI_DETECTION | FACE | ALERT | TRAINING | DOOR_OPEN | DOOR_CLOSE
        source_type:   upload | browser_camera | stream | video
        """
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
    # vision_compliance_hourly / vision_compliance_daily (upsert agregado)
    # =========================================================================

    async def upsert_compliance_hourly(
        self,
        company_id: int,
        hour_ts: str,           # "2025-03-01 14:00:00"
        total: int,
        compliant: int,
        zone_id: Optional[int] = None,
    ) -> bool:
        """uq_hour_zone: (company_id, hour_ts, zone_id)"""
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
        date: str,              # "2025-03-01"
        total: int,
        compliant: int,
        zone_id: Optional[int] = None,
    ) -> bool:
        """uq_day_zone: (company_id, date, zone_id)"""
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
        direction: str,         # edge_to_hub | hub_to_edge
        status: str,            # success | error | partial | skipped
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
        """
        Deleta anotações antigas da imagem e reinsere as novas.
        Colunas: company_id, image_name, class_id, class_name, cx, cy, w, h, source
        """
        try:
            # Remove anotações anteriores da mesma imagem
            await db.execute(
                "DELETE FROM vision_annotations WHERE company_id = %s AND image_name = %s",
                (company_id, image_name)
            )
            # Insere as novas
            for ann in annotations:
                cid = int(ann.get('class_id', ann.get('classId', 0)))
                await db.execute(
                    """
                    INSERT INTO vision_annotations
                        (company_id, image_name, class_id, class_name, cx, cy, w, h, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        company_id,
                        image_name,
                        cid,
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
        """Retorna anotações de uma imagem específica."""
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
        """Resumo de anotações: total de imagens e contagem por classe."""
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
    # LEITURA visionapp_* (gerenciados pelo Hub)
    # =========================================================================

    async def get_hub_people(self, company_id: int, active_only: bool = True) -> list:
        """visionapp_people — fonte de verdade do Hub."""
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
        """visionapp_recognition_events — eventos consolidados do Hub."""
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
        """
        Usa vision_compliance_hourly (pré-agregado) se disponível.
        Fallback: agrega em tempo real de vision_detection_events.
        """
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
        """
        Usa vision_epi_detections (eficiente).
        Fallback: parseia JSON de vision_detection_events.
        """
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


# Singleton — importe nos outros módulos com:
#   from app.core.repository import repo
repo = VisionRepository()