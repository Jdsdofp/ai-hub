"""
SmartX Vision Platform v3.0 — FastAPI Application Entry Point.
PPE Detection + Face Recognition + Annotation Converter + Live Streaming.

FIXES v3.1:
  - BUG-11: StaticFiles mount agora loga erro em vez de silenciosamente ignorar
"""
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from app.core.config import settings
from app.core.database import db
from app.core.company import CompanyData
from app.mqtt.client import mqtt_client
from app.streaming.manager import stream_manager




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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Working directory: {Path.cwd()}")
    CompanyData.init_base()
    try:
        await db.connect()
    except Exception as e:
        logger.warning(f"MySQL not available: {e}. Running without database.")
    try:
        await mqtt_client.connect()
    except Exception as e:
        logger.warning(f"MQTT not available: {e}. Running without MQTT.")
    yield
    stream_manager.stop_all()
    await mqtt_client.disconnect()
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

# API routes
from app.projects.epi_check.api.routes import router as epi_router
app.include_router(epi_router, prefix="/api/v1/epi")

# UI routes
from app.ui.routes import router as ui_router
app.include_router(ui_router, tags=["UI"])

# FILE routes
from app.projects.epi_check.api.filebrowser import router as fb_router
app.include_router(fb_router)


@app.get("/health", tags=["System"], summary="Health Check")
async def health():
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "mqtt": mqtt_client.is_connected,
        "edge_mode": settings.EDGE_MODE,
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
