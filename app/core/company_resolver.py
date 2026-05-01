"""
app/core/company_resolver.py
Resolve company_id -> CompanyInfo com cache TTL a partir do xfinderdb_prod.
"""
import time
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from app.core.xfinder_db import xfinder_db
from app.core.config import settings


@dataclass
class CompanyInfo:
    company_id: int
    full_name: Optional[str] = None
    web_site: Optional[str] = None
    admin_alias: Optional[str] = None
    def_city: Optional[str] = None
    def_state: Optional[str] = None
    def_country: Optional[str] = None
    def_region: Optional[str] = None
    def_phone1: Optional[str] = None
    def_notes: Optional[str] = None
    support_email: Optional[str] = None
    contact_email: Optional[str] = None
    financial_email: Optional[str] = None
    lang: str = "en"
    currency: Optional[str] = None
    time_zone: str = "UTC"
    logo: Optional[bytes] = None
    logo_small: Optional[bytes] = None
    image_type: Optional[str] = None
    license_volume: Optional[int] = None
    integration_key: Optional[str] = None
    branch_code: Optional[str] = None
    affiliate_code: Optional[str] = None
    flexible_fields: dict = field(default_factory=dict)

    @classmethod
    def from_db_row(cls, row: dict) -> "CompanyInfo":
        flex = {}
        for i in range(1, 21):
            val = row.get(f"flexible_field_name{i}")
            if val:
                flex[i] = val
        return cls(
            company_id=row["company_id"],
            full_name=row.get("full_name"),
            web_site=row.get("web_site"),
            admin_alias=row.get("admin_alias"),
            def_city=row.get("def_city"),
            def_state=row.get("def_state"),
            def_country=row.get("def_country"),
            def_region=row.get("def_region"),
            def_phone1=row.get("def_phone1"),
            def_notes=row.get("def_notes"),
            support_email=row.get("support_email"),
            contact_email=row.get("contact_email"),
            financial_email=row.get("financial_email"),
            lang=row.get("lang") or "en",
            currency=row.get("currency"),
            time_zone=row.get("time_zone") or "UTC",
            logo=row.get("logo"),
            logo_small=row.get("logo_small"),
            image_type=row.get("image_type"),
            license_volume=row.get("license_volume"),
            integration_key=row.get("integration_key"),
            branch_code=row.get("branch_code"),
            affiliate_code=row.get("affiliate_code"),
            flexible_fields=flex,
        )

    def to_safe_dict(self) -> dict:
        return {
            "company_id": self.company_id,
            "full_name": self.full_name,
            "web_site": self.web_site,
            "admin_alias": self.admin_alias,
            "city": self.def_city,
            "state": self.def_state,
            "country": self.def_country,
            "region": self.def_region,
            "lang": self.lang,
            "currency": self.currency,
            "time_zone": self.time_zone,
            "has_logo": self.logo is not None,
            "has_logo_small": self.logo_small is not None,
            "license_volume": self.license_volume,
            "branch_code": self.branch_code,
            "affiliate_code": self.affiliate_code,
            "flexible_fields": self.flexible_fields,
        }


class CompanyResolver:
    def __init__(self):
        self._cache: dict[int, tuple[Optional[CompanyInfo], float]] = {}

    @property
    def _ttl(self) -> int:
        return getattr(settings, "COMPANY_CACHE_TTL", 300)

    def _is_cached(self, company_id: int) -> bool:
        if company_id not in self._cache:
            return False
        _, ts = self._cache[company_id]
        if self._ttl <= 0:
            return True
        return (time.time() - ts) < self._ttl

    async def get(self, company_id: int) -> Optional[CompanyInfo]:
        if self._is_cached(company_id):
            info, _ = self._cache[company_id]
            return info

        if not xfinder_db.available:
            return None

        try:
            row = await xfinder_db.fetch_one(
                """
                SELECT
                    cd.company_id, cd.full_name, cd.web_site, cd.admin_alias,
                    cd.def_city, cd.def_state, cd.def_country, cd.def_region,
                    cd.def_phone1, cd.def_notes,
                    cd.support_email, cd.contact_email, cd.financial_email,
                    cd.lang, cd.currency, cd.time_zone,
                    cd.logo, cd.logo_small, cd.image_type,
                    cd.license_volume, cd.integration_key,
                    cd.branch_code, cd.affiliate_code,
                    cd.flexible_field_name1,  cd.flexible_field_name2,
                    cd.flexible_field_name3,  cd.flexible_field_name4,
                    cd.flexible_field_name5,  cd.flexible_field_name6,
                    cd.flexible_field_name7,  cd.flexible_field_name8,
                    cd.flexible_field_name9,  cd.flexible_field_name10,
                    cd.flexible_field_name11, cd.flexible_field_name12,
                    cd.flexible_field_name13, cd.flexible_field_name14,
                    cd.flexible_field_name15, cd.flexible_field_name16,
                    cd.flexible_field_name17, cd.flexible_field_name18,
                    cd.flexible_field_name19, cd.flexible_field_name20
                FROM company_details cd
                WHERE cd.company_id = %s
                LIMIT 1
                """,
                (company_id,),
            )
        except Exception as e:
            logger.warning(f"[CompanyResolver] DB error for company {company_id}: {e}")
            return None

        info = CompanyInfo.from_db_row(row) if row else None
        self._cache[company_id] = (info, time.time())

        if info:
            logger.info(f"[CompanyResolver] Resolved company {company_id}: {info.full_name} | {info.lang} | {info.time_zone}")
        else:
            logger.warning(f"[CompanyResolver] company_id={company_id} not found in xfinderdb")

        return info

    async def validate(self, company_id: int) -> bool:
        info = await self.get(company_id)
        return info is not None

    def invalidate(self, company_id: int):
        self._cache.pop(company_id, None)

    def cache_stats(self) -> dict:
        now = time.time()
        return {
            "cached_companies": list(self._cache.keys()),
            "ttl_seconds": self._ttl,
            "entries": [
                {
                    "company_id": cid,
                    "found": info is not None,
                    "age_seconds": round(now - ts),
                    "expires_in": max(0, round(self._ttl - (now - ts))) if self._ttl > 0 else "inf",
                }
                for cid, (info, ts) in self._cache.items()
            ],
        }


company_resolver = CompanyResolver()
