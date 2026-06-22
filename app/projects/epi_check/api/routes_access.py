"""
app/projects/epi_check/api/routes_access.py
============================================
Rotas REST para o módulo de Controle de Acesso (NR-36 / EPI Gate).
Monta em: /api/v1/epi/access
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

from app.core.repository_access import access_repo

router = APIRouter(prefix="/access", tags=["access-control"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class CameraUpsert(BaseModel):
    camera_code: str
    camera_name: Optional[str] = None
    site_id: Optional[int] = None
    zone_id: Optional[int] = None
    rtsp_url: Optional[str] = None
    api_base_url: Optional[str] = None


class DoorUpsert(BaseModel):
    door_code: str
    door_name: Optional[str] = None
    zone_id: Optional[int] = None
    camera_id: Optional[int] = None
    sensor_type: str = "GPIO"


class DoorEventCreate(BaseModel):
    door_code: str
    door_action: str
    door_id: Optional[int] = None
    camera_id: Optional[int] = None
    zone_id: Optional[int] = None
    person_code: Optional[str] = None
    person_name: Optional[str] = None
    obj_id: Optional[int] = None
    session_id: Optional[int] = None
    snapshot_path: Optional[str] = None
    snapshot_quality: Optional[float] = None
    exit_face_detected: bool = False
    exit_face_person_code: Optional[str] = None
    exit_face_confidence: Optional[float] = None
    exit_face_snapshot: Optional[str] = None
    exit_face_match: Optional[bool] = None
    exposure_session_id: Optional[int] = None
    exposure_minutes: Optional[float] = None
    sensor_type: str = "GPIO"
    authorized: Optional[bool] = None


class ExposureOpen(BaseModel):
    person_code: str
    person_name: Optional[str] = None
    obj_id: Optional[int] = None
    zone_id: Optional[int] = None
    door_id: Optional[int] = None
    door_code: Optional[str] = None
    daily_limit_min: float = 100.0
    entry_validation_id: Optional[int] = None


class ExposureClose(BaseModel):
    status: str = "COMPLETED"
    exit_door_event_id: Optional[int] = None
    exit_face_confirmed: Optional[bool] = None


class AccessRuleUpsert(BaseModel):
    rule_code: str
    rule_name: Optional[str] = None
    rule_type: str = "ppe_check"
    zone_id: Optional[int] = None
    terminal_id: Optional[int] = None
    max_minutes: int = 120
    warning_pct: int = 80
    window_type: str = "day"
    enforcement: str = "alert_and_supervisor"
    required_ppe: Optional[list] = None
    priority: int = 100


class ZoneRuleUpsert(BaseModel):
    zone_id: int
    ppe_config_id: int
    required: bool = True
    min_quantity: Optional[int] = None


class RecognitionEventCreate(BaseModel):
    camera_id: Optional[int] = None
    site_id: Optional[int] = None
    zone_id: Optional[int] = None
    person_code: Optional[str] = None
    person_name: Optional[str] = None
    face_recognized: bool = False
    face_confidence: Optional[float] = None
    face_snapshot_path: Optional[str] = None
    epi_coat: Optional[bool] = None
    epi_pants: Optional[bool] = None
    epi_gloves: Optional[bool] = None
    epi_cap: Optional[bool] = None
    epi_boots: Optional[bool] = None
    epi_total_required: int = 5
    epi_total_detected: int = 0
    epi_compliant: bool = False
    epi_units_required: int = 0
    epi_units_detected: int = 0
    epi_units_missing: int = 0
    compliance_score: Optional[float] = None
    compliance_detail: Optional[dict] = None
    exposure_accumulated_min: float = 0
    exposure_limit_min: float = 100
    exposure_exceeded: bool = False
    exposure_remaining_min: Optional[float] = None
    schedule_allowed: Optional[bool] = None
    schedule_id: Optional[int] = None
    access_decision: str = "GRANTED"
    denial_reason: Optional[str] = None
    door_command_sent: bool = False
    processing_time_ms: Optional[int] = None
    edge_device_id: Optional[int] = None
    vision_device_id: Optional[str] = None
    source: str = "SMARTX_VISION"
    validation_session_id: Optional[int] = None
    photo_seq: Optional[int] = None


# ── Câmeras ───────────────────────────────────────────────────────────────────

@router.get("/cameras")
async def list_cameras(
    company_id: int = Query(...),
    zone_id: Optional[int] = Query(None),
    active_only: bool = Query(True),
):
    return await access_repo.list_cameras(company_id, zone_id, active_only)


@router.post("/cameras", status_code=201)
async def upsert_camera(body: CameraUpsert, company_id: int = Query(...)):
    cam_id = await access_repo.upsert_camera(company_id=company_id, **body.model_dump())
    if cam_id is None:
        raise HTTPException(500, "Erro ao salvar câmera")
    return {"id": cam_id, "camera_code": body.camera_code}


@router.post("/cameras/{camera_code}/heartbeat")
async def camera_heartbeat(camera_code: str, company_id: int = Query(...)):
    return {"ok": await access_repo.camera_heartbeat(company_id, camera_code)}


# ── Portas ────────────────────────────────────────────────────────────────────

@router.get("/doors")
async def list_doors(
    company_id: int = Query(...),
    zone_id: Optional[int] = Query(None),
    active_only: bool = Query(True),
):
    return await access_repo.list_doors(company_id, zone_id, active_only)


@router.post("/doors", status_code=201)
async def upsert_door(body: DoorUpsert, company_id: int = Query(...)):
    door_id = await access_repo.upsert_door(company_id=company_id, **body.model_dump())
    if door_id is None:
        raise HTTPException(500, "Erro ao salvar porta")
    return {"id": door_id, "door_code": body.door_code}


# ── Door Events ───────────────────────────────────────────────────────────────

@router.post("/door-events", status_code=201)
async def create_door_event(body: DoorEventCreate, company_id: int = Query(...)):
    if body.door_action not in ("OPEN", "CLOSE"):
        raise HTTPException(400, "door_action deve ser OPEN ou CLOSE")
    event_id = await access_repo.save_door_event(company_id=company_id, **body.model_dump())
    if event_id is None:
        raise HTTPException(500, "Erro ao registrar evento de porta")
    return {"id": event_id, "door_action": body.door_action}


@router.get("/door-events")
async def list_door_events(
    company_id: int = Query(...),
    door_code: Optional[str] = Query(None),
    zone_id: Optional[int] = Query(None),
    person_code: Optional[str] = Query(None),
    hours: int = Query(24),
    limit: int = Query(100),
):
    return await access_repo.list_door_events(
        company_id, door_code, zone_id, person_code, hours, limit
    )


# ── Exposição NR-36 ───────────────────────────────────────────────────────────

@router.post("/exposure/open", status_code=201)
async def open_exposure(body: ExposureOpen, company_id: int = Query(...)):
    session_id = await access_repo.open_exposure_session(
        company_id=company_id, **body.model_dump()
    )
    if session_id is None:
        raise HTTPException(500, "Erro ao abrir sessão de exposição")
    return {"id": session_id, "person_code": body.person_code}


@router.post("/exposure/{exposure_id}/close")
async def close_exposure(
    exposure_id: int, body: ExposureClose, company_id: int = Query(...)
):
    result = await access_repo.close_exposure_session(
        company_id=company_id, exposure_session_id=exposure_id, **body.model_dump()
    )
    if result is None:
        raise HTTPException(404, "Sessão de exposição não encontrada")
    return result


@router.get("/exposure/persons-inside")
async def persons_inside(
    company_id: int = Query(...), zone_id: Optional[int] = Query(None)
):
    return await access_repo.get_persons_inside(company_id, zone_id)


@router.get("/exposure/daily/{person_code}")
async def daily_exposure(
    person_code: str,
    company_id: int = Query(...),
    date: Optional[str] = Query(None),
):
    return await access_repo.get_daily_exposure(company_id, person_code, date)


@router.get("/exposure/{exposure_id}/alerts")
async def exposure_alerts(exposure_id: int, company_id: int = Query(...)):
    return await access_repo.check_exposure_alerts(company_id, exposure_id)


@router.get("/exposure")
async def list_exposure(
    company_id: int = Query(...),
    zone_id: Optional[int] = Query(None),
    person_code: Optional[str] = Query(None),
    active_only: bool = Query(False),
    hours: int = Query(24),
    limit: int = Query(100),
):
    return await access_repo.list_exposure_sessions(
        company_id, zone_id, person_code, active_only, hours, limit
    )


# ── Regras de Acesso ──────────────────────────────────────────────────────────

@router.get("/access-rules")
async def get_access_rules(
    company_id: int = Query(...),
    zone_id: Optional[int] = Query(None),
    terminal_id: Optional[int] = Query(None),
    rule_type: Optional[str] = Query(None),
):
    return await access_repo.get_access_rules(company_id, zone_id, terminal_id, rule_type)


@router.post("/access-rules", status_code=201)
async def upsert_access_rule(body: AccessRuleUpsert, company_id: int = Query(...)):
    rule_id = await access_repo.upsert_access_rule(company_id=company_id, **body.model_dump())
    if rule_id is None:
        raise HTTPException(500, "Erro ao salvar regra")
    return {"id": rule_id, "rule_code": body.rule_code}


# ── Regras de EPI por Zona ────────────────────────────────────────────────────

@router.get("/zone-rules/{zone_id}")
async def get_zone_rules(zone_id: int, company_id: int = Query(...)):
    return await access_repo.get_zone_ppe_rules(company_id, zone_id)


@router.post("/zone-rules", status_code=201)
async def upsert_zone_rule(body: ZoneRuleUpsert, company_id: int = Query(...)):
    ok = await access_repo.upsert_zone_rule(company_id=company_id, **body.model_dump())
    if not ok:
        raise HTTPException(500, "Erro ao salvar regra de zona")
    return {"ok": True}


# ── Recognition Events ────────────────────────────────────────────────────────

@router.post("/recognition-events", status_code=201)
async def save_recognition_event(
    body: RecognitionEventCreate, company_id: int = Query(...)
):
    event_id = await access_repo.save_recognition_event(
        company_id=company_id, **body.model_dump()
    )
    if event_id is None:
        raise HTTPException(500, "Erro ao salvar recognition event")
    return {"id": event_id, "access_decision": body.access_decision}


# ── Dashboard ─────────────────────────────────────────────────────────────────

@router.get("/dashboard")
async def access_dashboard(
    company_id: int = Query(...), zone_id: Optional[int] = Query(None)
):
    return await access_repo.get_access_dashboard(company_id, zone_id)
