# """
# File Browser API — app/projects/epi_check/api/filebrowser.py
# Adicionar em main.py: from app.projects.epi_check.api.filebrowser import router as fb_router
#                        app.include_router(fb_router)
# """

# import os
# import shutil
# import mimetypes
# from pathlib import Path
# from datetime import datetime
# from typing import Optional

# from fastapi import APIRouter, HTTPException, Query
# from fastapi.responses import FileResponse, JSONResponse

# router = APIRouter(prefix="/api/v1/files", tags=["filebrowser"])

# # Raiz que será exposta — apenas a pasta data/ do projeto
# DATA_ROOT = Path("data")

# # Extensões permitidas para preview inline
# IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
# ALLOWED_DL  = IMAGE_EXTS | {".pt", ".yaml", ".txt", ".json", ".log", ".csv", ".mp4", ".avi"}


# def _safe_path(company_id: int, rel: str) -> Path:
#     """Resolve path dentro de data/{company_id}/ e garante que não sai da raiz."""
#     base = (DATA_ROOT / str(company_id)).resolve()
#     target = (base / rel.lstrip("/")).resolve()
#     # Proteção contra path traversal
#     if not str(target).startswith(str(base)):
#         raise HTTPException(400, "Caminho inválido")
#     return target


# def _file_info(path: Path, base: Path) -> dict:
#     stat = path.stat()
#     rel  = str(path.relative_to(base)).replace("\\", "/")
#     ext  = path.suffix.lower()
#     return {
#         "name":      path.name,
#         "path":      rel,
#         "is_dir":    path.is_dir(),
#         "size":      stat.st_size if path.is_file() else 0,
#         "size_fmt":  _fmt_size(stat.st_size) if path.is_file() else "—",
#         "modified":  datetime.fromtimestamp(stat.st_mtime).strftime("%d/%m/%Y %H:%M"),
#         "ext":       ext,
#         "is_image":  ext in IMAGE_EXTS,
#         "is_model":  ext == ".pt",
#         "mime":      mimetypes.guess_type(path.name)[0] or "application/octet-stream",
#     }


# def _fmt_size(b: int) -> str:
#     for unit in ("B", "KB", "MB", "GB"):
#         if b < 1024:
#             return f"{b:.1f} {unit}"
#         b /= 1024
#     return f"{b:.1f} TB"


# # ── LIST ──────────────────────────────────────────────────────────────────────
# @router.get("/list")
# def list_dir(
#     company_id: int = Query(1),
#     path: str = Query("", description="Caminho relativo dentro de data/{company_id}/"),
# ):
#     base   = (DATA_ROOT / str(company_id)).resolve()
#     target = _safe_path(company_id, path)

#     if not target.exists():
#         # Cria a pasta raiz da company se não existir
#         if path == "":
#             target.mkdir(parents=True, exist_ok=True)
#         else:
#             raise HTTPException(404, f"Pasta não encontrada: {path}")

#     if not target.is_dir():
#         raise HTTPException(400, "O caminho não é uma pasta")

#     items = []
#     try:
#         for p in sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
#             try:
#                 items.append(_file_info(p, base))
#             except (PermissionError, OSError):
#                 pass
#     except PermissionError:
#         raise HTTPException(403, "Sem permissão para listar esta pasta")

#     # Breadcrumb
#     parts = path.strip("/").split("/") if path.strip("/") else []
#     breadcrumb = [{"label": f"Company {company_id}", "path": ""}]
#     acc = ""
#     for part in parts:
#         acc = f"{acc}/{part}".lstrip("/")
#         breadcrumb.append({"label": part, "path": acc})

#     return {
#         "path":        path,
#         "breadcrumb":  breadcrumb,
#         "items":       items,
#         "total_items": len(items),
#         "dirs":        sum(1 for i in items if i["is_dir"]),
#         "files":       sum(1 for i in items if not i["is_dir"]),
#     }


# # ── DOWNLOAD / SERVE ──────────────────────────────────────────────────────────
# @router.get("/download")
# def download_file(
#     company_id: int = Query(1),
#     path: str = Query(...),
#     inline: bool = Query(False, description="True = abrir no browser (preview)"),
# ):
#     target = _safe_path(company_id, path)

#     if not target.exists() or not target.is_file():
#         raise HTTPException(404, "Arquivo não encontrado")

#     ext = target.suffix.lower()
#     if ext not in ALLOWED_DL:
#         raise HTTPException(403, f"Tipo de arquivo não permitido para download: {ext}")

#     disposition = "inline" if inline and ext in IMAGE_EXTS else "attachment"
#     return FileResponse(
#         path=target,
#         filename=target.name,
#         media_type=mimetypes.guess_type(target.name)[0] or "application/octet-stream",
#         headers={"Content-Disposition": f'{disposition}; filename="{target.name}"'},
#     )


# # ── THUMBNAIL (imagem redimensionada para preview rápido) ─────────────────────
# @router.get("/thumb")
# def get_thumbnail(
#     company_id: int = Query(1),
#     path: str = Query(...),
# ):
#     """Retorna a imagem original — o frontend faz o resize via CSS."""
#     return download_file(company_id=company_id, path=path, inline=True)


# # ── DELETE ────────────────────────────────────────────────────────────────────
# @router.delete("/delete")
# def delete_file(
#     company_id: int = Query(1),
#     path: str = Query(...),
# ):
#     target = _safe_path(company_id, path)

#     if not target.exists():
#         raise HTTPException(404, "Arquivo/pasta não encontrado")

#     # Proteção: não deixa deletar a raiz da company
#     base = (DATA_ROOT / str(company_id)).resolve()
#     if target == base:
#         raise HTTPException(400, "Não é possível deletar a pasta raiz")

#     try:
#         if target.is_dir():
#             shutil.rmtree(target)
#         else:
#             target.unlink()
#     except PermissionError:
#         raise HTTPException(403, "Sem permissão para deletar")

#     return {"deleted": True, "path": path}


# # ── STATS ─────────────────────────────────────────────────────────────────────
# @router.get("/stats")
# def folder_stats(company_id: int = Query(1)):
#     base = (DATA_ROOT / str(company_id)).resolve()
#     if not base.exists():
#         return {"total_size": 0, "total_size_fmt": "0 B", "file_count": 0, "image_count": 0, "model_count": 0}

#     total_size = 0
#     file_count = image_count = model_count = 0

#     for p in base.rglob("*"):
#         if p.is_file():
#             try:
#                 s = p.stat().st_size
#                 total_size  += s
#                 file_count  += 1
#                 ext = p.suffix.lower()
#                 if ext in IMAGE_EXTS: image_count += 1
#                 if ext == ".pt":      model_count  += 1
#             except OSError:
#                 pass

#     return {
#         "total_size":     total_size,
#         "total_size_fmt": _fmt_size(total_size),
#         "file_count":     file_count,
#         "image_count":    image_count,
#         "model_count":    model_count,
#     }



"""
File Browser API — app/projects/epi_check/api/filebrowser.py
Adicionar em main.py: from app.projects.epi_check.api.filebrowser import router as fb_router
                       app.include_router(fb_router)
"""

import os
import shutil
import mimetypes
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter(prefix="/api/v1/files", tags=["filebrowser"])

# Raiz que será exposta — apenas a pasta data/ do projeto
DATA_ROOT = Path("data")

# Extensões permitidas para preview inline
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
ALLOWED_DL  = IMAGE_EXTS | {".pt", ".yaml", ".txt", ".json", ".log", ".csv", ".mp4", ".avi"}


def _safe_path(company_id: int, rel: str) -> Path:
    """Resolve path dentro de data/{company_id}/ e garante que não sai da raiz."""
    base = (DATA_ROOT / str(company_id)).resolve()
    target = (base / rel.lstrip("/")).resolve()
    # Proteção contra path traversal
    if not str(target).startswith(str(base)):
        raise HTTPException(400, "Caminho inválido")
    return target


def _file_info(path: Path, base: Path) -> dict:
    stat = path.stat()
    rel  = str(path.relative_to(base)).replace("\\", "/")
    ext  = path.suffix.lower()
    return {
        "name":      path.name,
        "path":      rel,
        "is_dir":    path.is_dir(),
        "size":      stat.st_size if path.is_file() else 0,
        "size_fmt":  _fmt_size(stat.st_size) if path.is_file() else "—",
        "modified":  datetime.fromtimestamp(stat.st_mtime).strftime("%d/%m/%Y %H:%M"),
        "ext":       ext,
        "is_image":  ext in IMAGE_EXTS,
        "is_model":  ext == ".pt",
        "mime":      mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    }


def _fmt_size(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


# ── LIST ──────────────────────────────────────────────────────────────────────
@router.get("/list")
def list_dir(
    company_id: int = Query(1),
    path: str = Query("", description="Caminho relativo dentro de data/{company_id}/"),
):
    base   = (DATA_ROOT / str(company_id)).resolve()
    target = _safe_path(company_id, path)

    if not target.exists():
        # Cria a pasta raiz da company se não existir
        if path == "":
            target.mkdir(parents=True, exist_ok=True)
        else:
            raise HTTPException(404, f"Pasta não encontrada: {path}")

    if not target.is_dir():
        raise HTTPException(400, "O caminho não é uma pasta")

    items = []
    try:
        for p in sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            try:
                items.append(_file_info(p, base))
            except (PermissionError, OSError):
                pass
    except PermissionError:
        raise HTTPException(403, "Sem permissão para listar esta pasta")

    # Breadcrumb
    parts = path.strip("/").split("/") if path.strip("/") else []
    breadcrumb = [{"label": f"Company {company_id}", "path": ""}]
    acc = ""
    for part in parts:
        acc = f"{acc}/{part}".lstrip("/")
        breadcrumb.append({"label": part, "path": acc})

    return {
        "path":        path,
        "breadcrumb":  breadcrumb,
        "items":       items,
        "total_items": len(items),
        "dirs":        sum(1 for i in items if i["is_dir"]),
        "files":       sum(1 for i in items if not i["is_dir"]),
    }


# ── DOWNLOAD / SERVE ──────────────────────────────────────────────────────────
@router.get("/download")
def download_file(
    company_id: int = Query(1),
    path: str = Query(...),
    inline: bool = Query(False, description="True = abrir no browser (preview)"),
):
    target = _safe_path(company_id, path)

    if not target.exists() or not target.is_file():
        raise HTTPException(404, "Arquivo não encontrado")

    ext = target.suffix.lower()
    if ext not in ALLOWED_DL:
        raise HTTPException(403, f"Tipo de arquivo não permitido para download: {ext}")

    disposition = "inline" if inline and ext in IMAGE_EXTS else "attachment"
    return FileResponse(
        path=target,
        filename=target.name,
        media_type=mimetypes.guess_type(target.name)[0] or "application/octet-stream",
        headers={"Content-Disposition": f'{disposition}; filename="{target.name}"'},
    )


# ── THUMBNAIL (imagem redimensionada para preview rápido) ─────────────────────
@router.get("/thumb")
def get_thumbnail(
    company_id: int = Query(1),
    path: str = Query(...),
):
    """Retorna a imagem original — o frontend faz o resize via CSS."""
    return download_file(company_id=company_id, path=path, inline=True)


# ── DELETE ────────────────────────────────────────────────────────────────────
@router.delete("/delete")
def delete_file(
    company_id: int = Query(1),
    path: str = Query(...),
):
    target = _safe_path(company_id, path)

    if not target.exists():
        raise HTTPException(404, "Arquivo/pasta não encontrado")

    # Proteção: não deixa deletar a raiz da company
    base = (DATA_ROOT / str(company_id)).resolve()
    if target == base:
        raise HTTPException(400, "Não é possível deletar a pasta raiz")

    try:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    except PermissionError:
        raise HTTPException(403, "Sem permissão para deletar")

    return {"deleted": True, "path": path}


# ── STATS ─────────────────────────────────────────────────────────────────────
@router.get("/stats")
def folder_stats(company_id: int = Query(1)):
    base = (DATA_ROOT / str(company_id)).resolve()
    if not base.exists():
        return {"total_size": 0, "total_size_fmt": "0 B", "file_count": 0, "image_count": 0, "model_count": 0}

    total_size = 0
    file_count = image_count = model_count = 0

    for p in base.rglob("*"):
        if p.is_file():
            try:
                s = p.stat().st_size
                total_size  += s
                file_count  += 1
                ext = p.suffix.lower()
                if ext in IMAGE_EXTS: image_count += 1
                if ext == ".pt":      model_count  += 1
            except OSError:
                pass

    return {
        "total_size":     total_size,
        "total_size_fmt": _fmt_size(total_size),
        "file_count":     file_count,
        "image_count":    image_count,
        "model_count":    model_count,
    }


# ── UPLOAD ────────────────────────────────────────────────────────────────────
from fastapi import UploadFile, File as FastAPIFile
from typing import List

UPLOAD_MAX_MB = 500  # limite por arquivo

@router.post("/upload")
async def upload_files(
    company_id: int = Query(1),
    path: str = Query("", description="Pasta destino relativa a data/{company_id}/"),
    files: List[UploadFile] = FastAPIFile(...),
):
    """Faz upload de um ou mais arquivos para a pasta indicada."""
    dest_dir = _safe_path(company_id, path)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not dest_dir.is_dir():
        raise HTTPException(400, "O destino não é uma pasta")

    results = []
    errors  = []

    for upload in files:
        # Valida tamanho
        content = await upload.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > UPLOAD_MAX_MB:
            errors.append({"name": upload.filename, "error": f"Arquivo muito grande ({size_mb:.1f} MB > {UPLOAD_MAX_MB} MB)"})
            continue

        # Evita sobrescrever: se já existe, adiciona sufixo
        dest_file = dest_dir / upload.filename
        if dest_file.exists():
            stem = dest_file.stem
            ext  = dest_file.suffix
            counter = 1
            while dest_file.exists():
                dest_file = dest_dir / f"{stem}_{counter}{ext}"
                counter += 1

        dest_file.write_bytes(content)
        results.append({
            "name":      upload.filename,
            "saved_as":  dest_file.name,
            "size_fmt":  _fmt_size(len(content)),
            "path":      str(dest_file.relative_to((DATA_ROOT / str(company_id)).resolve())).replace("\\", "/"),
        })

    return {
        "uploaded": len(results),
        "errors":   len(errors),
        "files":    results,
        "error_details": errors,
    }


# ── MKDIR ─────────────────────────────────────────────────────────────────────
@router.post("/mkdir")
def make_dir(
    company_id: int = Query(1),
    path: str = Query(..., description="Novo caminho relativo (ex: fotos/novos)"),
):
    target = _safe_path(company_id, path)
    if target.exists():
        raise HTTPException(400, "Pasta já existe")
    target.mkdir(parents=True, exist_ok=False)
    return {"created": True, "path": path}