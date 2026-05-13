"""
SmartX Vision Platform v3.0 — FastAPI Application Entry Point.
PPE Detection + Face Recognition + Annotation Converter + Live Streaming.

FIXES v3.1:
  - BUG-11: StaticFiles mount agora loga erro em vez de silenciosamente ignorar
"""
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.core.database import db
from app.core.xfinder_db import xfinder_db
from app.core.company import CompanyData
from app.mqtt.client import mqtt_client
from app.streaming.manager import stream_manager

# No início do main.py, antes de qualquer import
# from app.core.config import settings
# print("HOST:", settings.MYSQL_HOST)  # deve mostrar aihubdb.smartxhub.io


OPENAPI_TAGS = [
    {
        "name": "PPE Configuration",
        "description": "Manage which PPE classes (helmet, boots, gloves, etc.) are active per company. "
                       "Only enabled classes are used in training, detection, and compliance checks.",
    },
    {
        "name": "Photo Upload",
        "description": "Upload training images organized by PPE category, or bulk upload images with "
                       "pre-existing YOLO TXT annotation files.",
    },
    {
        "name": "Annotation Converter",
        "description": "Convert Roboflow polygon/OBB annotations (9 values per line) to standard "
                       "YOLO bounding box format (5 values per line). Supports class ID remapping.",
    },
    {
        "name": "Visual Annotation",
        "description": "Browser-based annotation tool. List images, serve them for drawing bounding boxes, "
                       "load/save YOLO labels, and check annotation statistics.",
    },
    {
        "name": "Dataset & Training",
        "description": "Generate YOLOv8 train/valid datasets from annotated images, start background training, "
                       "and monitor training progress. Supports yolov8n/s/m/l/x base models.",
    },
    {
        "name": "Detection — Image",
        "description": "Run PPE detection on single images via base64 (REST API) or file upload (UI). "
                       "Optionally includes face recognition with adjustable confidence thresholds.",
    },
    {
        "name": "Detection — Video",
        "description": "Process uploaded video files or YouTube URLs frame-by-frame for PPE compliance analysis. "
                       "Returns per-frame results and overall compliance rate.",
    },
    {
        "name": "Face Recognition",
        "description": "Register people with face photos and badge IDs. The system builds face embeddings using "
                       "InsightFace (buffalo_l) and matches faces during detection with cosine similarity.",
    },
    {
        "name": "Live Streaming",
        "description": "Real-time PPE detection on RTSP cameras, YouTube live streams, webcams, or video files. "
                       "Provides MJPEG feed for browser display and per-frame detection results.",
    },
    {
        "name": "Models & Results",
        "description": "List trained models, serve detection result snapshots, and view comprehensive "
                       "system statistics including storage, people count, and training status.",
    },
    {
        "name": "System",
        "description": "Health check and system status endpoints.",
    },
]


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
#     logger.info(f"Working directory: {Path.cwd()}")
#     CompanyData.init_base()
#     try:
#         await db.connect()
#     except Exception as e:
#         logger.warning(f"MySQL not available: {e}. Running without database.")
#     try:
#         await mqtt_client.connect()
#     except Exception as e:
#         logger.warning(f"MQTT not available: {e}. Running without MQTT.")
#     yield
#     stream_manager.stop_all()
#     await mqtt_client.disconnect()
#     await xfinder_db.disconnect()
#     logger.info("Shutdown complete")
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Working directory: {Path.cwd()}")
    CompanyData.init_base()

    try:
        await db.connect()
        await xfinder_db.connect()
    except Exception as e:
        logger.warning(f"MySQL not available: {e}. Running without database.")

    try:
        await mqtt_client.connect()
    except Exception as e:
        logger.warning(f"MQTT not available: {e}. Running without MQTT.")

    # Pré-carrega PPE config do banco para o cache do engine
    try:
        from app.core.repository import repo
        from app.projects.epi_check.engine.detector import epi_engine
        db_companies = await db.fetch_all("SELECT DISTINCT company_id FROM vision_ppe_config", ())
        if not db_companies:
            db_companies = [{"company_id": 1}]
        for row in db_companies:
            cid = row["company_id"]
            cfg = await repo.get_ppe_config(cid)
            if cfg:
                epi_engine.set_ppe_config_cache(cid, cfg)
                logger.info(f"[Company {cid}] PPE config loaded from DB")
            else:
                logger.warning(f"[Company {cid}] No PPE config in DB — using DEFAULT")
    except Exception as e:
        logger.warning(f"PPE config preload failed: {e}. Engine will use DEFAULT.")

    yield

    stream_manager.stop_all()
    await mqtt_client.disconnect()
    await xfinder_db.disconnect()
    await db.disconnect()
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "## SmartX Vision Platform v3\n\n"
        "AI-powered **PPE (Personal Protective Equipment) detection** with "
        "**face recognition**, **annotation tools**, and **live streaming**.\n\n"
        "### Key Features\n"
        "- **YOLOv8** object detection for 6 PPE classes (helmet, boots, gloves, coat, pants, person)\n"
        "- **InsightFace** face recognition with per-company people registry\n"
        "- **Roboflow converter** — polygon/OBB → YOLO bbox format\n"
        "- **Visual annotation** — draw bounding boxes in the browser\n"
        "- **Live streaming** — RTSP, YouTube, webcam with real-time detection\n"
        "- **MQTT integration** — alerts to SmartX HUB\n"
        "- **Multi-company** — full data isolation per company_id\n\n"
        "### Authentication\n"
        "All endpoints require `company_id` via query parameter or `X-Company-ID` header.\n"
        "Secured endpoints also require `X-API-Key` header.\n\n"
        "### PPE Class IDs\n"
        "| ID | Class | Default |\n"
        "|---|---|---|\n"
        "| 0 | thermal_coat | Disabled |\n"
        "| 1 | thermal_pants | Disabled |\n"
        "| 2 | gloves | Disabled |\n"
        "| 3 | helmet | **Enabled** |\n"
        "| 4 | boots | **Enabled** |\n"
        "| 5 | person | **Enabled** |\n"
    ),
    openapi_tags=OPENAPI_TAGS,
    contact={
        "name": "SmartX Technology Inc.",
        "url": "https://smartx.com",
        "email": "support@smartx.com",
    },
    license_info={
        "name": "Proprietary",
    },
    lifespan=lifespan,
)

# FIX BUG-11: Static files com log de erro em vez de falha silenciosa
static_dir = Path("app/ui/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"Static files mounted from: {static_dir.resolve()}")
else:
    logger.warning(
        f"Static files directory not found: {static_dir.resolve()}. "
        "Logo e assets da UI não serão servidos. "
        "Crie a pasta app/ui/static/img/ e adicione o logoSmartx.jpeg."
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://analyticsapp.smartxhub.cloud",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





@app.get("/companies", tags=["System"])
async def list_companies():
    """Lista todas as empresas do xfinderdb_prod."""
    from app.core.xfinder_db import xfinder_db
    if not xfinder_db.available:
        raise HTTPException(503, detail="XFinder DB not available")
    rows = await xfinder_db.fetch_all(
        """
        SELECT cd.company_id, cd.full_name, cd.admin_alias,
               cd.def_city, cd.def_country, cd.lang, cd.time_zone
        FROM company_details cd
        INNER JOIN company c ON c.id = cd.company_id
        ORDER BY cd.company_id ASC
        """,
        ()
    )
    return [
        {
            "company_id": r["company_id"],
            "full_name": r["full_name"] or f"Company {r['company_id']}",
            "admin_alias": r["admin_alias"],
            "city": r["def_city"],
            "country": r["def_country"],
            "lang": r["lang"],
            "time_zone": r["time_zone"],
        }
        for r in rows
    ]




# ── Monitor endpoint ─────────────────────────────────────────────────────────
@app.get("/monitor", tags=["System"])
async def get_monitor_stats(company_id: int = 1):
    """Painel de monitoramento — disco, GPU, CPU, RAM, keys, detecções."""
    import os, shutil, glob
    from app.core.xfinder_db import xfinder_db

    stats = {}

    # ── Disco ────────────────────────────────────────────────────────────────
    try:
        total, used, free = shutil.disk_usage("/opt/vision/data")
        data_dir = "/opt/vision/data"
        debug_count = len(glob.glob(f"{data_dir}/debug_*.jpg"))
        debug_size  = sum(os.path.getsize(f) for f in glob.glob(f"{data_dir}/debug_*.jpg") if os.path.exists(f))

        # Tamanho por company
        company_dirs = []
        for d in os.listdir(data_dir):
            full = os.path.join(data_dir, d)
            if os.path.isdir(full) and d.isdigit():
                size = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, files in os.walk(full)
                    for f in files
                )
                company_dirs.append({"company_id": int(d), "size_mb": round(size/1024/1024, 2)})
        company_dirs.sort(key=lambda x: x["size_mb"], reverse=True)

        stats["disk"] = {
            "total_gb":   round(total/1024**3, 2),
            "used_gb":    round(used/1024**3, 2),
            "free_gb":    round(free/1024**3, 2),
            "used_pct":   round(used/total*100, 1),
            "data_size_mb": round(sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, files in os.walk(data_dir)
                for f in files
            )/1024/1024, 2),
            "debug_count": debug_count,
            "debug_size_kb": round(debug_size/1024, 1),
            "company_dirs": company_dirs[:10],
        }
    except Exception as e:
        stats["disk"] = {"error": str(e)}

    # ── CPU / RAM ─────────────────────────────────────────────────────────────
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        stats["system"] = {
            "cpu_pct":    round(cpu, 1),
            "ram_total_gb": round(mem.total/1024**3, 2),
            "ram_used_gb":  round(mem.used/1024**3, 2),
            "ram_pct":      round(mem.percent, 1),
        }
    except Exception as e:
        stats["system"] = {"error": str(e)}

    # ── GPU ───────────────────────────────────────────────────────────────────
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], timeout=3
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        stats["gpu"] = {
            "name":       parts[0],
            "util_pct":   int(parts[1]),
            "mem_used_mb": int(parts[2]),
            "mem_total_mb": int(parts[3]),
            "mem_pct":    round(int(parts[2])/int(parts[3])*100, 1),
            "temp_c":     int(parts[4]),
        }
    except Exception as e:
        stats["gpu"] = {"error": str(e)}

    # ── API Keys ─────────────────────────────────────────────────────────────
    try:
        if xfinder_db.available:
            rows = await xfinder_db.fetch_all(
                """SELECT
                    COUNT(*) as total,
                    SUM(active=1 AND (valid_until IS NULL OR valid_until >= CURDATE()) AND revoked_at IS NULL) as active_count,
                    SUM(revoked_at IS NOT NULL) as revoked_count,
                    SUM(valid_until IS NOT NULL AND valid_until < CURDATE()) as expired_count,
                    SUM(use_count) as total_uses,
                    MAX(last_used_at) as last_used
                FROM vision_api_keys WHERE company_id = %s""",
                (company_id,)
            )
            r = rows[0] if rows else {}
            stats["api_keys"] = {
                "total":         int(r.get("total") or 0),
                "active":        int(r.get("active_count") or 0),
                "revoked":       int(r.get("revoked_count") or 0),
                "expired":       int(r.get("expired_count") or 0),
                "total_uses":    int(r.get("total_uses") or 0),
                "last_used":     str(r.get("last_used") or ""),
            }
        else:
            stats["api_keys"] = {"error": "xfinder_db unavailable"}
    except Exception as e:
        stats["api_keys"] = {"error": str(e)}

    # ── Detecções ────────────────────────────────────────────────────────────
    try:
        from app.core.database import db
        rows = await db.fetch_all(
            """SELECT
                COUNT(*) as total_events,
                SUM(compliant=1) as compliant,
                SUM(compliant=0) as non_compliant,
                ROUND(AVG(compliance_score)*100,1) as avg_compliance_pct,
                MAX(created_at) as last_event
            FROM vision_detection_events
            WHERE company_id = %s AND created_at >= NOW() - INTERVAL 24 HOUR""",
            (company_id,)
        )
        r = rows[0] if rows else {}

        # Últimos 7 dias por dia
        daily = await db.fetch_all(
            """SELECT DATE(created_at) as day,
                COUNT(*) as events,
                ROUND(AVG(compliance_score)*100,1) as compliance_pct
            FROM vision_detection_events
            WHERE company_id = %s AND created_at >= NOW() - INTERVAL 7 DAY
            GROUP BY DATE(created_at) ORDER BY day ASC""",
            (company_id,)
        )

        stats["detections"] = {
            "last_24h": {
                "total":          int(r.get("total_events") or 0),
                "compliant":      int(r.get("compliant") or 0),
                "non_compliant":  int(r.get("non_compliant") or 0),
                "avg_compliance": float(r.get("avg_compliance_pct") or 0),
                "last_event":     str(r.get("last_event") or ""),
            },
            "daily_7d": [
                {"day": str(d["day"]), "events": int(d["events"]), "compliance_pct": float(d["compliance_pct"] or 0)}
                for d in daily
            ]
        }
    except Exception as e:
        stats["detections"] = {"error": str(e)}

    return stats

# ── Station Init endpoint ─────────────────────────────────────────────────────
@app.get("/station/init", tags=["System"])
async def station_init(x_api_key: str = Header(None, alias="X-API-Key")):
    """
    Valida token da máquina e retorna tema + dados da empresa.
    Chamado pelo EpiCameraStation na inicialização.
    """
    from app.core.api_key_service import api_key_service as _aks2
    from app.core.xfinder_db import xfinder_db

    if not x_api_key:
        raise HTTPException(401, detail="X-API-Key header required")

    info = await _aks2.validate(x_api_key)
    if not info:
        raise HTTPException(401, detail="Invalid, expired or inactive API key")

    if not xfinder_db.available:
        raise HTTPException(503, detail="XFinder DB not available")

    # Busca tema da empresa
    theme_row = await xfinder_db.fetch_one(
        "SELECT * FROM company_theme WHERE company_id = %s LIMIT 1",
        (info.company_id,)
    )

    # Busca dados da empresa
    company_row = await xfinder_db.fetch_one(
        """SELECT full_name, admin_alias, logo, logo_small, image_type,
                  lang, time_zone, currency, def_city, def_country
           FROM company_details WHERE company_id = %s LIMIT 1""",
        (info.company_id,)
    )

    # Monta logo em base64 se existir
    import base64
    logo_b64 = None
    logo_type = "image/png"
    if company_row and company_row.get("logo"):
        raw = company_row["logo"]
        if isinstance(raw, (bytes, bytearray)):
            # Decodifica bytes para string primeiro
            raw_str = raw.decode("ascii", errors="ignore").strip()
            img_type = company_row.get("image_type") or "image/png"
            if img_type.startswith("data:"):
                img_type = img_type.split(":")[1].split(";")[0].strip()
            if "/" not in img_type:
                img_type = "image/png"
            logo_type = img_type
            # Se já é base64 puro (começa com iVBOR ou similar), usa direto
            # Se é binário real (começa com bytes não-ASCII), então encoda
            if all(c < 128 for c in raw[:10]):
                logo_b64 = raw_str  # já é base64 como texto
            else:
                logo_b64 = base64.b64encode(raw).decode()
        elif isinstance(raw, str):
            # String — pode ser data URL completa ex: data:image/png;base64,iVBOR...
            s = raw.strip()
            if s.startswith("data:"):
                try:
                    header, b64 = s.split(",", 1)
                    logo_type = header.split(":")[1].split(";")[0]
                    logo_b64 = b64  # já é base64 puro, usar direto
                except Exception:
                    logo_b64 = None
            else:
                logo_b64 = s  # base64 puro sem prefixo

    theme = {}
    if theme_row:
        def hex_color(v):
            if not v: return None
            v = str(v).strip().lstrip("#")
            return f"#{v}" if v else None
        theme = {
            "colorPrimary":          hex_color(theme_row.get("color_primary"))          or "#2373FE",
            "colorPrimaryDark":      hex_color(theme_row.get("color_primary_dark"))      or "#919297",
            "colorAccent":           hex_color(theme_row.get("color_accent"))            or "#239ed3",
            "colorFontTitle":        hex_color(theme_row.get("color_font_title"))        or "#FFFFFF",
            "colorFontSubtitle":     hex_color(theme_row.get("color_font_subtitle"))     or "#FFFFFF",
            "colorMenuBackground":   hex_color(theme_row.get("color_webmenu_background")) or "#2373FE",
            "colorLogoBackground":   hex_color(theme_row.get("color_weblogo_background")) or "#2373FE",
            "colorHeaderBg":         hex_color(theme_row.get("color_header_bg"))         or "#2373FE",
            "colorSidebarActive":    hex_color(theme_row.get("color_sidebar_active"))    or "#2373FE",
            "logoUrl":               theme_row.get("logo_url"),
        }

    return {
        "valid": True,
        "key": info.to_dict(),
        "company": {
            "company_id":  info.company_id,
            "full_name":   company_row.get("full_name")   if company_row else info.company_name,
            "admin_alias": company_row.get("admin_alias") if company_row else None,
            "lang":        company_row.get("lang")        if company_row else "en",
            "time_zone":   company_row.get("time_zone")   if company_row else "UTC",
            "currency":    company_row.get("currency")    if company_row else None,
            "city":        company_row.get("def_city")    if company_row else None,
            "country":     company_row.get("def_country") if company_row else None,
            "logo":        (f"data:{logo_type};base64,{logo_b64}" if logo_b64 else None),
        },
        "theme": theme,
    }

# ── API Keys endpoints ────────────────────────────────────────────────────────
from app.core.api_key_service import api_key_service as _aks
from fastapi import Query as _Query, Header as _Header
from typing import Optional as _Optional
from datetime import date as _date

@app.post("/api-keys", tags=["API Keys"])
async def create_api_key(body: dict):
    try:
        result = await _aks.create(
            company_id     = body["company_id"],
            machine_id     = body["machine_id"],
            machine_name   = body.get("machine_name"),
            device_profile = body.get("device_profile", "Generic"),
            module         = body.get("module", "epi_station"),
            source         = body.get("source", "manual"),
            location       = body.get("location"),
            site_id        = body.get("site_id"),
            zone_id        = body.get("zone_id"),
            license_type   = body.get("license_type", "trial"),
            license_volume = body.get("license_volume", 1),
            valid_from     = _date.fromisoformat(body["valid_from"])  if body.get("valid_from")  else None,
            valid_until    = _date.fromisoformat(body["valid_until"]) if body.get("valid_until") else None,
            rate_limit_rpm = body.get("rate_limit_rpm", 120),
            description    = body.get("description"),
            created_by     = body.get("created_by"),
        )
        return {"success": True, "message": "Copie o token agora — nao sera exibido novamente.", "data": result}
    except Exception as e:
        raise HTTPException(400, detail=str(e))

@app.get("/api-keys", tags=["API Keys"])
async def list_api_keys(company_id: _Optional[int] = _Query(None)):
    if company_id:
        rows = await _aks.list_by_company(company_id)
    else:
        rows = await _aks.list_all()
    return {"success": True, "data": rows, "total": len(rows)}

@app.get("/api-keys/validate", tags=["API Keys"])
async def validate_api_key(x_api_key: str = _Header(None, alias="X-API-Key")):
    if not x_api_key:
        raise HTTPException(401, detail="X-API-Key header required")
    info = await _aks.validate(x_api_key)
    if not info:
        raise HTTPException(401, detail="Invalid, expired or inactive API key")
    return {"valid": True, "status": info.status, "data": info.to_dict()}

@app.delete("/api-keys/{key_id}", tags=["API Keys"])
async def revoke_api_key(key_id: int, reason: str = _Query(None)):
    await _aks.revoke(key_id, reason)
    return {"success": True, "revoked": key_id}

@app.delete("/api-keys/{key_id}/permanent", tags=["API Keys"])
async def delete_api_key_permanent(key_id: int):
    """Exclui permanentemente uma API key do banco."""
    from app.core.xfinder_db import xfinder_db
    if not xfinder_db.available:
        raise HTTPException(503, detail="XFinder DB not available")
    await xfinder_db.fetch_one(
        "DELETE FROM vision_api_keys WHERE id = %s",
        (key_id,)
    )
    return {"success": True, "deleted": key_id}

@app.patch("/api-keys/{key_id}/toggle", tags=["API Keys"])
async def toggle_api_key(key_id: int, active: bool = _Query(...)):
    await _aks.toggle_active(key_id, active)
    return {"success": True, "key_id": key_id, "active": active}


# ── Company endpoints (xfinderdb_prod) ───────────────────────────────────────
from app.core.company_resolver import company_resolver

@app.get("/company/cache/stats", tags=["System"])
async def company_cache_stats():
    return company_resolver.cache_stats()

@app.delete("/company/{company_id}/cache", tags=["System"])
async def invalidate_company_cache(company_id: int):
    company_resolver.invalidate(company_id)
    return {"invalidated": company_id}

@app.get("/company/{company_id}", tags=["System"])
async def get_company_details(company_id: int):
    from app.core.xfinder_db import xfinder_db
    if not xfinder_db.available:
        raise HTTPException(503, detail="XFinder DB not available")
    info = await company_resolver.get(company_id)
    if not info:
        raise HTTPException(404, detail=f"Company {company_id} not found")
    return info.to_safe_dict()

@app.get("/company/{company_id}/logo", tags=["System"])
async def get_company_logo(company_id: int, small: bool = False):
    from fastapi.responses import Response
    from app.core.xfinder_db import xfinder_db
    if not xfinder_db.available:
        raise HTTPException(503, detail="XFinder DB not available")
    info = await company_resolver.get(company_id)
    if not info:
        raise HTTPException(404, detail=f"Company {company_id} not found")
    logo_bytes = info.logo_small if small else info.logo
    if not logo_bytes:
        raise HTTPException(404, detail="Logo not found for this company")
    return Response(content=logo_bytes, media_type=info.image_type or "image/jpeg")

# API routes
from app.projects.epi_check.api.routes import router as epi_router
app.include_router(epi_router, prefix="/api/v1/epi")

from app.projects.epi_check.api.ws_proxy import router as ws_proxy_router
app.include_router(ws_proxy_router, prefix="/api/v1/epi")

from app.projects.epi_check.api.ws_epi_stream import router as ws_epi_stream_router
app.include_router(ws_epi_stream_router, prefix="/api/v1/epi")

# UI routes
from app.ui.routes import router as ui_router
app.include_router(ui_router, tags=["UI"])

# FILE routes
from app.projects.epi_check.api.filebrowser import router as fb_router
app.include_router(fb_router)


@app.get("/health", tags=["System"])
async def health():
    # Teste direto do banco
    try:
        row = await db.fetch_one("SELECT 1 AS ok")
        db_status = "connected" if row else "no response"
    except Exception as e:
        db_status = f"ERROR: {e}"

    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "mqtt": mqtt_client.is_connected,
        "db": db_status,          # ← mostra o status real
        "db_pool": str(db._pool), # ← mostra se o pool existe
    }

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        
        # Permissões para câmera e microfone
        response.headers["Permissions-Policy"] = (
            "camera=(self \"https://*.cloudflare.com\" \"https://*.google.com\" \"https://*.yourdomain.com\" *), "
            "microphone=(), "
            "geolocation=(), "
            "interest-cohort=()"
        )
        
        # Feature-Policy (legado, mas ainda usado por alguns browsers)
        response.headers["Feature-Policy"] = (
            "camera 'self' https://*.cloudflare.com https://*.google.com https://*.yourdomain.com *; "
            "microphone 'none'"
        )
        
        # CSP (Content-Security-Policy) - Importante para permitir media devices
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://*.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "media-src 'self' blob:; "
            "connect-src 'self' https://*.cloudflare.com wss://*.cloudflare.com; "
            "frame-ancestors 'none';"
        )
        
        # Headers adicionais para Cloudflare
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response

# Substitua o middleware anterior por este
app.add_middleware(SecurityHeadersMiddleware)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        workers=settings.APP_WORKERS,
        reload=settings.APP_DEBUG,
        log_level=settings.APP_LOG_LEVEL.lower(),
    )
