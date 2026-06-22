"""
VisionRepository — Extensão: Controle de Acesso
================================================
Cobre as tabelas que ainda não estão em repository.py:

  ESCRITA
    vision_cameras              — cadastro de câmeras
    vision_doors                — cadastro de portas/catracas
    vision_door_events          — eventos OPEN/CLOSE com snapshot e face de saída
    vision_exposure_sessions    — tempo de exposição NR-36 por pessoa/zona
    vision_access_rules         — regras de acesso (NR-36, EPI, horário, combinadas)
    vision_zone_rules           — quais EPIs são exigidos por zona
    visionapp_recognition_events — evento completo de reconhecimento (Hub-side write)

  LEITURA
    Todas as tabelas acima + joins para dashboards de acesso

Uso:
    from app.core.repository_access import access_repo

    # No ws_epi_stream, após a decisão:
    session_db_id = await access_repo.create_validation_session(...)
    await access_repo.add_validation_photo(...)
    await access_repo.close_validation_session(...)
    await access_repo.open_door_event(...)
    await access_repo.open_exposure_session(...)
    await access_repo.save_recognition_event(...)
"""

import json
from datetime import datetime, date
from typing import Optional

from loguru import logger

from app.core.database import db


class AccessRepository:

    # =========================================================================
    # vision_cameras
    # =========================================================================

    async def upsert_camera(
        self,
        company_id: int,
        camera_code: str,
        camera_name: Optional[str] = None,
        site_id: Optional[int] = None,
        zone_id: Optional[int] = None,
        rtsp_url: Optional[str] = None,
        api_base_url: Optional[str] = None,
    ) -> Optional[int]:
        """Cria ou atualiza câmera. Retorna id."""
        try:
            await db.execute(
                """
                INSERT INTO vision_cameras
                    (company_id, camera_code, camera_name, site_id, zone_id,
                     rtsp_url, api_base_url, active, last_heartbeat)
                VALUES (%s,%s,%s,%s,%s,%s,%s, 1, NOW())
                ON DUPLICATE KEY UPDATE
                    camera_name   = COALESCE(VALUES(camera_name),   camera_name),
                    site_id       = COALESCE(VALUES(site_id),       site_id),
                    zone_id       = COALESCE(VALUES(zone_id),       zone_id),
                    rtsp_url      = COALESCE(VALUES(rtsp_url),      rtsp_url),
                    api_base_url  = COALESCE(VALUES(api_base_url),  api_base_url),
                    last_heartbeat = NOW()
                """,
                (company_id, camera_code, camera_name, site_id, zone_id,
                 rtsp_url, api_base_url),
            )
            row = await db.fetch_one(
                "SELECT id FROM vision_cameras WHERE company_id=%s AND camera_code=%s",
                (company_id, camera_code),
            )
            return row["id"] if row else None
        except Exception as e:
            logger.error(f"[AccessRepo] upsert_camera: {e}")
            return None

    async def camera_heartbeat(self, company_id: int, camera_code: str) -> bool:
        try:
            await db.execute(
                "UPDATE vision_cameras SET last_heartbeat=NOW() "
                "WHERE company_id=%s AND camera_code=%s",
                (company_id, camera_code),
            )
            return True
        except Exception as e:
            logger.warning(f"[AccessRepo] camera_heartbeat: {e}")
            return False

    async def list_cameras(
        self, company_id: int, zone_id: Optional[int] = None, active_only: bool = True
    ) -> list:
        try:
            conds = ["company_id = %s"]
            params: list = [company_id]
            if zone_id is not None:
                conds.append("zone_id = %s")
                params.append(zone_id)
            if active_only:
                conds.append("active = 1")
            where = " AND ".join(conds)
            rows = await db.fetch_all(
                f"SELECT *, TIMESTAMPDIFF(SECOND, last_heartbeat, NOW()) AS secs_since_hb "
                f"FROM vision_cameras WHERE {where} ORDER BY camera_name",
                params,
            )
            now = datetime.utcnow()
            result = []
            for r in rows:
                d = dict(r)
                hb = d.get("last_heartbeat")
                d["online"] = (now - hb).total_seconds() < 120 if hb else False
                result.append(d)
            return result
        except Exception as e:
            logger.error(f"[AccessRepo] list_cameras: {e}")
            return []

    # =========================================================================
    # vision_doors
    # =========================================================================

    async def upsert_door(
        self,
        company_id: int,
        door_code: str,
        door_name: Optional[str] = None,
        zone_id: Optional[int] = None,
        camera_id: Optional[int] = None,
        sensor_type: str = "GPIO",
    ) -> Optional[int]:
        """Cria ou atualiza porta/catraca. Retorna id."""
        try:
            await db.execute(
                """
                INSERT INTO vision_doors
                    (company_id, door_code, door_name, zone_id, camera_id, sensor_type, active)
                VALUES (%s,%s,%s,%s,%s,%s, 1)
                ON DUPLICATE KEY UPDATE
                    door_name   = COALESCE(VALUES(door_name),  door_name),
                    zone_id     = COALESCE(VALUES(zone_id),    zone_id),
                    camera_id   = COALESCE(VALUES(camera_id),  camera_id),
                    sensor_type = VALUES(sensor_type)
                """,
                (company_id, door_code, door_name, zone_id, camera_id, sensor_type),
            )
            row = await db.fetch_one(
                "SELECT id FROM vision_doors WHERE company_id=%s AND door_code=%s",
                (company_id, door_code),
            )
            return row["id"] if row else None
        except Exception as e:
            logger.error(f"[AccessRepo] upsert_door: {e}")
            return None

    async def get_door(
        self, company_id: int, door_code: str
    ) -> Optional[dict]:
        try:
            return await db.fetch_one(
                "SELECT * FROM vision_doors WHERE company_id=%s AND door_code=%s",
                (company_id, door_code),
            )
        except Exception as e:
            logger.error(f"[AccessRepo] get_door: {e}")
            return None

    async def list_doors(
        self, company_id: int, zone_id: Optional[int] = None, active_only: bool = True
    ) -> list:
        try:
            conds = ["d.company_id = %s"]
            params: list = [company_id]
            if zone_id is not None:
                conds.append("d.zone_id = %s")
                params.append(zone_id)
            if active_only:
                conds.append("d.active = 1")
            where = " AND ".join(conds)
            return await db.fetch_all(
                f"""
                SELECT d.*, c.camera_name
                FROM vision_doors d
                LEFT JOIN vision_cameras c ON c.id = d.camera_id
                WHERE {where}
                ORDER BY d.door_name
                """,
                params,
            )
        except Exception as e:
            logger.error(f"[AccessRepo] list_doors: {e}")
            return []

    # =========================================================================
    # vision_door_events
    # =========================================================================

    async def save_door_event(
        self,
        company_id: int,
        door_code: str,
        door_action: str,
        door_id: Optional[int] = None,
        camera_id: Optional[int] = None,
        zone_id: Optional[int] = None,
        person_code: Optional[str] = None,
        person_name: Optional[str] = None,
        obj_id: Optional[int] = None,
        session_id: Optional[int] = None,
        snapshot_path: Optional[str] = None,
        snapshot_quality: Optional[float] = None,
        exit_face_detected: bool = False,
        exit_face_person_code: Optional[str] = None,
        exit_face_confidence: Optional[float] = None,
        exit_face_snapshot: Optional[str] = None,
        exit_face_match: Optional[bool] = None,
        exposure_session_id: Optional[int] = None,
        exposure_minutes: Optional[float] = None,
        sensor_type: str = "GPIO",
        authorized: Optional[bool] = None,
    ) -> Optional[int]:
        try:
            return await db.insert_get_id(
                """
                INSERT INTO vision_door_events (
                    company_id, door_id, door_code, camera_id, current_zone_id,
                    door_action, obj_id, person_code, person_name, session_id,
                    snapshot_path, snapshot_captured_at, snapshot_quality,
                    exit_face_detected, exit_face_person_code, exit_face_confidence,
                    exit_face_snapshot, exit_face_match,
                    exposure_session_id, exposure_minutes,
                    sensor_type, authorized
                ) VALUES (
                    %s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,
                    %s, NOW(), %s,
                    %s,%s,%s,
                    %s,%s,
                    %s,%s,
                    %s,%s
                )
                """,
                (
                    company_id, door_id, door_code, camera_id, zone_id,
                    door_action, obj_id, person_code, person_name, session_id,
                    snapshot_path, snapshot_quality,
                    exit_face_detected, exit_face_person_code, exit_face_confidence,
                    exit_face_snapshot, exit_face_match,
                    exposure_session_id, exposure_minutes,
                    sensor_type, authorized,
                ),
            )
        except Exception as e:
            logger.error(f"[AccessRepo] save_door_event: {e}")
            return None

    async def get_last_door_event(
        self,
        company_id: int,
        door_code: str,
        action: Optional[str] = None,
    ) -> Optional[dict]:
        try:
            extra = "AND door_action = %s" if action else ""
            params = (company_id, door_code, action) if action else (company_id, door_code)
            return await db.fetch_one(
                f"""
                SELECT * FROM vision_door_events
                WHERE company_id = %s AND door_code = %s {extra}
                ORDER BY event_timestamp DESC LIMIT 1
                """,
                params,
            )
        except Exception as e:
            logger.error(f"[AccessRepo] get_last_door_event: {e}")
            return None

    async def list_door_events(
        self,
        company_id: int,
        door_code: Optional[str] = None,
        zone_id: Optional[int] = None,
        person_code: Optional[str] = None,
        hours: int = 24,
        limit: int = 100,
    ) -> list:
        try:
            conds = ["company_id = %s", "event_timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)"]
            params: list = [company_id, hours]
            if door_code:
                conds.append("door_code = %s")
                params.append(door_code)
            if zone_id is not None:
                conds.append("current_zone_id = %s")
                params.append(zone_id)
            if person_code:
                conds.append("person_code = %s")
                params.append(person_code)
            params.append(limit)
            where = " AND ".join(conds)
            return await db.fetch_all(
                f"""
                SELECT id, event_timestamp, door_code, door_action,
                       person_code, person_name, authorized,
                       exit_face_detected, exit_face_match,
                       exposure_minutes, snapshot_path, current_zone_id
                FROM vision_door_events
                WHERE {where}
                ORDER BY event_timestamp DESC LIMIT %s
                """,
                params,
            )
        except Exception as e:
            logger.error(f"[AccessRepo] list_door_events: {e}")
            return []

    # =========================================================================
    # vision_exposure_sessions
    # =========================================================================

    async def open_exposure_session(
        self,
        company_id: int,
        person_code: str,
        person_name: Optional[str] = None,
        obj_id: Optional[int] = None,
        zone_id: Optional[int] = None,
        door_id: Optional[int] = None,
        door_code: Optional[str] = None,
        daily_limit_min: float = 100.0,
        entry_validation_id: Optional[int] = None,
    ) -> Optional[int]:
        try:
            acc_row = await db.fetch_one(
                """
                SELECT COALESCE(SUM(duration_minutes), 0) AS accumulated
                FROM vision_exposure_sessions
                WHERE company_id = %s AND person_code = %s
                  AND session_status IN ('COMPLETED','TIMEOUT','FORCED_CLOSE')
                  AND DATE(entry_timestamp) = CURDATE()
                """,
                (company_id, person_code),
            )
            accumulated = float((acc_row or {}).get("accumulated", 0))
            remaining = max(daily_limit_min - accumulated, 0)

            return await db.insert_get_id(
                """
                INSERT INTO vision_exposure_sessions (
                    company_id, obj_id, person_code, person_name,
                    current_zone_id, door_id, door_code,
                    entry_timestamp, session_status,
                    daily_accumulated_min, daily_limit_min, daily_remaining_min,
                    limit_exceeded, entry_validation_id
                ) VALUES (
                    %s,%s,%s,%s,
                    %s,%s,%s,
                    NOW(), 'ACTIVE',
                    %s,%s,%s,
                    %s,%s
                )
                """,
                (
                    company_id, obj_id, person_code, person_name,
                    zone_id, door_id, door_code,
                    accumulated, daily_limit_min, remaining,
                    accumulated >= daily_limit_min,
                    entry_validation_id,
                ),
            )
        except Exception as e:
            logger.error(f"[AccessRepo] open_exposure_session: {e}")
            return None

    async def close_exposure_session(
        self,
        company_id: int,
        exposure_session_id: int,
        status: str = "COMPLETED",
        exit_door_event_id: Optional[int] = None,
        exit_face_confirmed: Optional[bool] = None,
    ) -> Optional[dict]:
        try:
            row = await db.fetch_one(
                "SELECT * FROM vision_exposure_sessions WHERE id = %s AND company_id = %s",
                (exposure_session_id, company_id),
            )
            if not row:
                return None

            await db.execute(
                """
                UPDATE vision_exposure_sessions SET
                    exit_timestamp      = NOW(),
                    duration_seconds    = TIMESTAMPDIFF(SECOND, entry_timestamp, NOW()),
                    duration_minutes    = ROUND(TIMESTAMPDIFF(SECOND, entry_timestamp, NOW()) / 60.0, 2),
                    session_status      = %s,
                    exit_door_event_id  = COALESCE(%s, exit_door_event_id),
                    exit_face_confirmed = COALESCE(%s, exit_face_confirmed)
                WHERE id = %s AND company_id = %s
                """,
                (status, exit_door_event_id, exit_face_confirmed,
                 exposure_session_id, company_id),
            )

            closed = await db.fetch_one(
                "SELECT duration_minutes, limit_exceeded, daily_accumulated_min, daily_limit_min "
                "FROM vision_exposure_sessions WHERE id = %s",
                (exposure_session_id,),
            )
            return dict(closed) if closed else None
        except Exception as e:
            logger.error(f"[AccessRepo] close_exposure_session: {e}")
            return None

    async def get_active_exposure_session(
        self, company_id: int, person_code: str
    ) -> Optional[dict]:
        try:
            return await db.fetch_one(
                """
                SELECT * FROM vision_exposure_sessions
                WHERE company_id = %s AND person_code = %s
                  AND session_status = 'ACTIVE'
                ORDER BY entry_timestamp DESC LIMIT 1
                """,
                (company_id, person_code),
            )
        except Exception as e:
            logger.error(f"[AccessRepo] get_active_exposure_session: {e}")
            return None

    async def get_daily_exposure(
        self, company_id: int, person_code: str, target_date: Optional[str] = None
    ) -> dict:
        try:
            date_cond = "DATE(entry_timestamp) = %s" if target_date else "DATE(entry_timestamp) = CURDATE()"
            params = (company_id, person_code, target_date) if target_date else (company_id, person_code)
            row = await db.fetch_one(
                f"""
                SELECT
                    COUNT(*)                       AS sessions_count,
                    COALESCE(SUM(CASE WHEN session_status IN ('COMPLETED','TIMEOUT','FORCED_CLOSE')
                                THEN duration_minutes ELSE 0 END), 0) AS completed_minutes,
                    COALESCE(SUM(CASE WHEN session_status = 'ACTIVE'
                                THEN TIMESTAMPDIFF(SECOND, entry_timestamp, NOW()) / 60.0
                                ELSE 0 END), 0)                        AS active_minutes,
                    MAX(daily_limit_min)           AS daily_limit_min,
                    MAX(limit_exceeded)            AS limit_exceeded
                FROM vision_exposure_sessions
                WHERE company_id = %s AND person_code = %s AND {date_cond}
                """,
                params,
            )
            if not row:
                return {"completed_minutes": 0, "active_minutes": 0,
                        "total_minutes": 0, "daily_limit_min": 100,
                        "limit_exceeded": False, "sessions_count": 0}
            total = float(row["completed_minutes"]) + float(row["active_minutes"])
            limit = float(row["daily_limit_min"] or 100)
            return {
                "completed_minutes": round(float(row["completed_minutes"]), 2),
                "active_minutes":    round(float(row["active_minutes"]), 2),
                "total_minutes":     round(total, 2),
                "remaining_minutes": round(max(limit - total, 0), 2),
                "daily_limit_min":   limit,
                "limit_exceeded":    total >= limit,
                "sessions_count":    int(row["sessions_count"]),
            }
        except Exception as e:
            logger.error(f"[AccessRepo] get_daily_exposure: {e}")
            return {}

    async def check_exposure_alerts(
        self, company_id: int, exposure_session_id: int
    ) -> dict:
        try:
            row = await db.fetch_one(
                """
                SELECT daily_accumulated_min, daily_limit_min,
                       alert_sent_50pct, alert_sent_80pct, alert_sent_limit,
                       TIMESTAMPDIFF(SECOND, entry_timestamp, NOW()) / 60.0 AS current_minutes
                FROM vision_exposure_sessions
                WHERE id = %s AND company_id = %s
                """,
                (exposure_session_id, company_id),
            )
            if not row:
                return {}

            accumulated = float(row["daily_accumulated_min"])
            current = float(row["current_minutes"])
            total = accumulated + current
            limit = float(row["daily_limit_min"])
            pct = total / limit if limit > 0 else 0

            alerts = {}
            updates = []

            if pct >= 0.5 and not row["alert_sent_50pct"]:
                alerts["alert_50pct"] = True
                updates.append("alert_sent_50pct = 1")
            if pct >= 0.8 and not row["alert_sent_80pct"]:
                alerts["alert_80pct"] = True
                updates.append("alert_sent_80pct = 1")
            if pct >= 1.0 and not row["alert_sent_limit"]:
                alerts["alert_limit"] = True
                updates.append("alert_sent_limit = 1")
                updates.append("limit_exceeded = 1")

            if updates:
                await db.execute(
                    f"UPDATE vision_exposure_sessions SET {', '.join(updates)} WHERE id = %s",
                    (exposure_session_id,),
                )

            alerts["total_minutes"] = round(total, 2)
            alerts["pct_used"] = round(pct * 100, 1)
            return alerts
        except Exception as e:
            logger.error(f"[AccessRepo] check_exposure_alerts: {e}")
            return {}

    async def list_exposure_sessions(
        self,
        company_id: int,
        zone_id: Optional[int] = None,
        person_code: Optional[str] = None,
        active_only: bool = False,
        hours: int = 24,
        limit: int = 100,
    ) -> list:
        try:
            conds = ["company_id = %s",
                     "entry_timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)"]
            params: list = [company_id, hours]
            if zone_id is not None:
                conds.append("current_zone_id = %s")
                params.append(zone_id)
            if person_code:
                conds.append("person_code = %s")
                params.append(person_code)
            if active_only:
                conds.append("session_status = 'ACTIVE'")
            params.append(limit)
            where = " AND ".join(conds)
            return await db.fetch_all(
                f"""
                SELECT id, person_code, person_name, current_zone_id, door_code,
                       entry_timestamp, exit_timestamp,
                       duration_minutes, session_status,
                       daily_accumulated_min, daily_limit_min, daily_remaining_min,
                       limit_exceeded, alert_sent_80pct, alert_sent_limit
                FROM vision_exposure_sessions
                WHERE {where}
                ORDER BY entry_timestamp DESC LIMIT %s
                """,
                params,
            )
        except Exception as e:
            logger.error(f"[AccessRepo] list_exposure_sessions: {e}")
            return []

    # =========================================================================
    # vision_access_rules
    # =========================================================================

    async def get_access_rules(
        self,
        company_id: int,
        zone_id: Optional[int] = None,
        terminal_id: Optional[int] = None,
        rule_type: Optional[str] = None,
    ) -> list:
        try:
            conds = ["company_id = %s", "active = 1"]
            params: list = [company_id]
            if zone_id is not None:
                conds.append("(zone_id = %s OR zone_id IS NULL)")
                params.append(zone_id)
            if terminal_id is not None:
                conds.append("(terminal_id = %s OR terminal_id IS NULL)")
                params.append(terminal_id)
            if rule_type:
                conds.append("rule_type = %s")
                params.append(rule_type)
            where = " AND ".join(conds)
            return await db.fetch_all(
                f"""
                SELECT id, rule_code, rule_name, rule_type,
                       zone_id, terminal_id,
                       max_minutes, warning_pct, window_type,
                       enforcement, required_ppe, priority
                FROM vision_access_rules
                WHERE {where}
                ORDER BY priority ASC
                """,
                params,
            )
        except Exception as e:
            logger.error(f"[AccessRepo] get_access_rules: {e}")
            return []

    async def upsert_access_rule(
        self,
        company_id: int,
        rule_code: str,
        rule_name: Optional[str] = None,
        rule_type: str = "ppe_check",
        zone_id: Optional[int] = None,
        terminal_id: Optional[int] = None,
        max_minutes: int = 120,
        warning_pct: int = 80,
        window_type: str = "day",
        enforcement: str = "alert_and_supervisor",
        required_ppe: Optional[list] = None,
        priority: int = 100,
    ) -> Optional[int]:
        try:
            await db.execute(
                """
                INSERT INTO vision_access_rules (
                    company_id, rule_code, rule_name, rule_type,
                    zone_id, terminal_id,
                    max_minutes, warning_pct, window_type,
                    enforcement, required_ppe, priority, active
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, 1)
                ON DUPLICATE KEY UPDATE
                    rule_name   = COALESCE(VALUES(rule_name),   rule_name),
                    rule_type   = VALUES(rule_type),
                    max_minutes = VALUES(max_minutes),
                    warning_pct = VALUES(warning_pct),
                    window_type = VALUES(window_type),
                    enforcement = VALUES(enforcement),
                    required_ppe = COALESCE(VALUES(required_ppe), required_ppe),
                    priority    = VALUES(priority),
                    active      = 1,
                    updated_at  = NOW()
                """,
                (
                    company_id, rule_code, rule_name, rule_type,
                    zone_id, terminal_id,
                    max_minutes, warning_pct, window_type,
                    enforcement,
                    json.dumps(required_ppe) if required_ppe else None,
                    priority,
                ),
            )
            row = await db.fetch_one(
                "SELECT id FROM vision_access_rules WHERE company_id=%s AND rule_code=%s",
                (company_id, rule_code),
            )
            return row["id"] if row else None
        except Exception as e:
            logger.error(f"[AccessRepo] upsert_access_rule: {e}")
            return None

    # =========================================================================
    # vision_zone_rules
    # =========================================================================

    async def get_zone_ppe_rules(
        self, company_id: int, zone_id: int
    ) -> list:
        try:
            return await db.fetch_all(
                """
                SELECT
                    zr.id, zr.zone_id, zr.required,
                    pc.class_name, pc.display_name, pc.body_region,
                    pc.confidence_min,
                    COALESCE(zr.min_quantity, pc.min_quantity) AS min_quantity
                FROM vision_zone_rules zr
                JOIN vision_ppe_config pc ON pc.id = zr.ppe_config_id
                WHERE zr.company_id = %s AND zr.zone_id = %s
                  AND zr.active = 1 AND pc.enabled = 1
                ORDER BY pc.body_region, pc.class_name
                """,
                (company_id, zone_id),
            )
        except Exception as e:
            logger.error(f"[AccessRepo] get_zone_ppe_rules: {e}")
            return []

    async def upsert_zone_rule(
        self,
        company_id: int,
        zone_id: int,
        ppe_config_id: int,
        required: bool = True,
        min_quantity: Optional[int] = None,
    ) -> bool:
        try:
            await db.execute(
                """
                INSERT INTO vision_zone_rules
                    (company_id, zone_id, ppe_config_id, required, min_quantity, active)
                VALUES (%s,%s,%s,%s,%s, 1)
                ON DUPLICATE KEY UPDATE
                    required     = VALUES(required),
                    min_quantity = VALUES(min_quantity),
                    active       = 1
                """,
                (company_id, zone_id, ppe_config_id, required, min_quantity),
            )
            return True
        except Exception as e:
            logger.error(f"[AccessRepo] upsert_zone_rule: {e}")
            return False

    # =========================================================================
    # visionapp_recognition_events
    # =========================================================================

    async def save_recognition_event(
        self,
        company_id: int,
        camera_id: Optional[int] = None,
        site_id: Optional[int] = None,
        zone_id: Optional[int] = None,
        person_code: Optional[str] = None,
        person_name: Optional[str] = None,
        face_recognized: bool = False,
        face_confidence: Optional[float] = None,
        face_snapshot_path: Optional[str] = None,
        epi_coat: Optional[bool] = None,
        epi_pants: Optional[bool] = None,
        epi_gloves: Optional[bool] = None,
        epi_cap: Optional[bool] = None,
        epi_boots: Optional[bool] = None,
        epi_total_required: int = 5,
        epi_total_detected: int = 0,
        epi_compliant: bool = False,
        epi_units_required: int = 0,
        epi_units_detected: int = 0,
        epi_units_missing: int = 0,
        compliance_score: Optional[float] = None,
        compliance_detail: Optional[dict] = None,
        exposure_accumulated_min: float = 0,
        exposure_limit_min: float = 100,
        exposure_exceeded: bool = False,
        exposure_remaining_min: Optional[float] = None,
        schedule_allowed: Optional[bool] = None,
        schedule_id: Optional[int] = None,
        access_decision: str = "GRANTED",
        denial_reason: Optional[str] = None,
        door_command_sent: bool = False,
        processing_time_ms: Optional[int] = None,
        edge_device_id: Optional[int] = None,
        vision_device_id: Optional[str] = None,
        source: str = "SMARTX_VISION",
        validation_session_id: Optional[int] = None,
        photo_seq: Optional[int] = None,
    ) -> Optional[int]:
        try:
            return await db.insert_get_id(
                """
                INSERT INTO visionapp_recognition_events (
                    company_id,
                    event_timestamp, camera_id, site_id, zone_id,
                    person_code, person_name,
                    face_recognized, face_confidence, face_snapshot_path,
                    epi_coat, epi_pants, epi_gloves, epi_cap, epi_boots,
                    epi_total_required, epi_total_detected, epi_compliant,
                    epi_units_required, epi_units_detected, epi_units_missing,
                    compliance_score, compliance_detail,
                    exposure_accumulated_min, exposure_limit_min,
                    exposure_exceeded, exposure_remaining_min,
                    schedule_allowed, schedule_id,
                    access_decision, denial_reason,
                    door_command_sent, processing_time_ms,
                    edge_device_id, vision_device_id, source,
                    validation_session_id, photo_seq
                ) VALUES (
                    %s,
                    NOW(), %s,%s,%s,
                    %s,%s,
                    %s,%s,%s,
                    %s,%s,%s,%s,%s,
                    %s,%s,%s,
                    %s,%s,%s,
                    %s,%s,
                    %s,%s,
                    %s,%s,
                    %s,%s,
                    %s,%s,
                    %s,%s,
                    %s,%s,%s,
                    %s,%s
                )
                """,
                (
                    company_id,
                    camera_id, site_id, zone_id,
                    person_code, person_name,
                    face_recognized, face_confidence, face_snapshot_path,
                    epi_coat, epi_pants, epi_gloves, epi_cap, epi_boots,
                    epi_total_required, epi_total_detected, epi_compliant,
                    epi_units_required, epi_units_detected, epi_units_missing,
                    compliance_score,
                    json.dumps(compliance_detail) if compliance_detail else None,
                    exposure_accumulated_min, exposure_limit_min,
                    exposure_exceeded, exposure_remaining_min,
                    schedule_allowed, schedule_id,
                    access_decision, denial_reason,
                    door_command_sent, processing_time_ms,
                    edge_device_id, vision_device_id, source,
                    validation_session_id, photo_seq,
                ),
            )
        except Exception as e:
            logger.error(f"[AccessRepo] save_recognition_event: {e}")
            return None

    # =========================================================================
    # ANALYTICS
    # =========================================================================

    async def get_access_dashboard(
        self, company_id: int, zone_id: Optional[int] = None
    ) -> dict:
        try:
            cond_zone = "AND current_zone_id = %s" if zone_id is not None else ""
            p_base = [company_id]
            if zone_id is not None:
                p_base.append(zone_id)

            active = await db.fetch_one(
                f"SELECT COUNT(*) AS cnt FROM vision_exposure_sessions "
                f"WHERE company_id = %s AND session_status = 'ACTIVE' {cond_zone}",
                p_base,
            )
            exceeded = await db.fetch_one(
                f"SELECT COUNT(DISTINCT person_code) AS cnt FROM vision_exposure_sessions "
                f"WHERE company_id = %s AND limit_exceeded = 1 "
                f"AND DATE(entry_timestamp) = CURDATE() {cond_zone}",
                p_base,
            )
            denied = await db.fetch_one(
                f"SELECT COUNT(*) AS cnt FROM vision_validation_sessions "
                f"WHERE company_id = %s AND access_decision NOT IN ('GRANTED','PENDING') "
                f"AND DATE(created_at) = CURDATE() "
                f"{'AND current_zone_id = %s' if zone_id else ''}",
                p_base,
            )
            entries = await db.fetch_one(
                f"SELECT COUNT(*) AS cnt FROM vision_door_events "
                f"WHERE company_id = %s AND door_action = 'OPEN' "
                f"AND DATE(event_timestamp) = CURDATE() {cond_zone}",
                p_base,
            )
            return {
                "persons_inside":    (active   or {}).get("cnt", 0),
                "exposure_exceeded": (exceeded or {}).get("cnt", 0),
                "access_denied":     (denied   or {}).get("cnt", 0),
                "door_opens_today":  (entries  or {}).get("cnt", 0),
            }
        except Exception as e:
            logger.error(f"[AccessRepo] get_access_dashboard: {e}")
            return {}

    async def get_persons_inside(
        self, company_id: int, zone_id: Optional[int] = None
    ) -> list:
        try:
            cond_zone = "AND es.current_zone_id = %s" if zone_id is not None else ""
            params = [company_id]
            if zone_id is not None:
                params.append(zone_id)
            return await db.fetch_all(
                f"""
                SELECT es.person_code, es.person_name,
                       es.current_zone_id, es.door_code,
                       es.entry_timestamp,
                       ROUND(TIMESTAMPDIFF(SECOND, es.entry_timestamp, NOW()) / 60.0, 1) AS minutes_inside,
                       es.daily_accumulated_min, es.daily_limit_min,
                       es.limit_exceeded,
                       p.badge_id, p.department
                FROM vision_exposure_sessions es
                LEFT JOIN vision_people p
                    ON p.company_id = es.company_id AND p.person_code = es.person_code
                WHERE es.company_id = %s AND es.session_status = 'ACTIVE' {cond_zone}
                ORDER BY es.entry_timestamp
                """,
                params,
            )
        except Exception as e:
            logger.error(f"[AccessRepo] get_persons_inside: {e}")
            return []


# Singleton
access_repo = AccessRepository()
