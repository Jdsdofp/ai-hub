"""
Company data isolation. Structure: data/{company_id}/{project}/...
"""
import os
from pathlib import Path
from app.core.config import settings


class CompanyData:
    PROJECTS = ["epi_check", "assets", "vehicles"]

    EPI_SUBDIRS = [
        "photos_raw", "annotations", "raw_labels",
        "dataset/train/images", "dataset/train/labels",
        "dataset/valid/images", "dataset/valid/labels",
        "models", "results", "snapshots",
        "faces/photos", "faces/embeddings", "people", "exports", "temp",
    ]

    @classmethod
    def init_base(cls):
        Path(settings.DATA_ROOT).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

    @classmethod
    def get_project_root(cls, company_id: int, project: str = "epi_check") -> Path:
        p = Path(settings.DATA_ROOT) / str(company_id) / project
        if not p.exists():
            cls._create_project_dirs(company_id, project)
        return p

    @classmethod
    def _create_project_dirs(cls, company_id: int, project: str):
        base = Path(settings.DATA_ROOT) / str(company_id) / project
        if project == "epi_check":
            for sub in cls.EPI_SUBDIRS:
                (base / sub).mkdir(parents=True, exist_ok=True)
        else:
            for sub in ["data", "models", "results", "temp"]:
                (base / sub).mkdir(parents=True, exist_ok=True)

    @classmethod
    def path(cls, company_id: int, project: str, *segments: str) -> Path:
        root = cls.get_project_root(company_id, project)
        p = root.joinpath(*segments)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def epi(cls, company_id: int, *segments: str) -> Path:
        return cls.path(company_id, "epi_check", *segments)

    @classmethod
    def disk_usage(cls, company_id: int) -> dict:
        root = Path(settings.DATA_ROOT) / str(company_id)
        total = 0
        count = 0
        for dp, dn, fnames in os.walk(root):
            for f in fnames:
                total += os.path.getsize(os.path.join(dp, f))
                count += 1
        return {"company_id": company_id, "total_mb": round(total / 1048576, 2), "files": count}
