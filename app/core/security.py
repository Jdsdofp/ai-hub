"""
API security: key validation and company_id extraction.
"""
from fastapi import Header, HTTPException, Depends, Query
from typing import Optional
from app.core.config import settings


async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")) -> str:
    if x_api_key and x_api_key == settings.API_KEY:
        return x_api_key
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def get_company_id(
    x_company_id: str = Header(None, alias="X-Company-ID"),
    company_id: Optional[int] = Query(None),
) -> int:
    cid = x_company_id or company_id
    if cid is None:
        raise HTTPException(status_code=400, detail="X-Company-ID header or company_id param required")
    try:
        cid_int = int(cid)
        if cid_int <= 0:
            raise ValueError
        return cid_int
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"Invalid company_id: '{cid}'")


async def get_authenticated_company(
    api_key: str = Depends(verify_api_key),
    company_id: int = Depends(get_company_id),
) -> int:
    return company_id


async def get_ui_company(
    company_id: Optional[int] = Query(None),
    x_company_id: str = Header(None, alias="X-Company-ID"),
) -> int:
    cid = company_id or x_company_id
    if cid is None:
        return 1
    try:
        return int(cid)
    except (ValueError, TypeError):
        return 1
