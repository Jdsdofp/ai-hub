"""
UI routes — serve the web dashboard.
"""
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.core.security import get_ui_company

templates = Jinja2Templates(directory="app/ui/templates")
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, company_id: int = Depends(get_ui_company)):
    return templates.TemplateResponse("dashboard.html", {
        "request": request, "company_id": company_id,
    })


