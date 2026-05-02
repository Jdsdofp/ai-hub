"""
app/core/api_key_service.py
Serviço de API Keys por máquina — vision_api_keys no xfinderdb_prod.
"""
import hashlib
import secrets
from dataclasses import dataclass
from datetime import date
from typing import Optional
from loguru import logger

from app.core.xfinder_db import xfinder_db


@dataclass
class ApiKeyInfo:
    id:             int
    company_id:     int
    company_name:   Optional[str]
    token:          str
    machine_id:     str
    machine_name:   Optional[str]
    device_profile: Optional[str]
    module:         str
    source:         Optional[str]
    location:       Optional[str]
    site_id:        Optional[int]
    zone_id:        Optional[int]
    license_type:   Optional[str]
    license_volume: int
    valid_from:     Optional[date]
    valid_until:    Optional[date]
    days_remaining: Optional[int]
    status:         str
    use_count:      int
    rate_limit_rpm: int

    @property
    def is_valid(self) -> bool:
        return self.status == 'ACTIVE'

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "company_id":     self.company_id,
            "company_name":   self.company_name,
            "machine_id":     self.machine_id,
            "machine_name":   self.machine_name,
            "device_profile": self.device_profile,
            "module":         self.module,
            "location":       self.location,
            "license_type":   self.license_type,
            "license_volume": self.license_volume,
            "valid_until":    str(self.valid_until) if self.valid_until else None,
            "days_remaining": self.days_remaining,
            "status":         self.status,
            "use_count":      self.use_count,
        }


class ApiKeyService:

    @staticmethod
    def generate_token() -> str:
        return secrets.token_hex(32)

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    async def create(
        self,
        company_id:     int,
        machine_id:     str,
        machine_name:   Optional[str]  = None,
        device_profile: str            = 'Generic',
        module:         str            = 'epi_station',
        source:         str            = 'manual',
        location:       Optional[str]  = None,
        site_id:        Optional[int]  = None,
        zone_id:        Optional[int]  = None,
        license_type:   str            = 'trial',
        license_volume: int            = 1,
        valid_from:     Optional[date] = None,
        valid_until:    Optional[date] = None,
        rate_limit_rpm: int            = 120,
        description:    Optional[str]  = None,
        created_by:     Optional[str]  = None,
    ) -> dict:
        if not xfinder_db.available:
            raise RuntimeError("xfinder_db not available")

        token = self.generate_token()
        hash_ = self.hash_token(token)

        async with xfinder_db._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO vision_api_keys
                      (company_id, token, hash, machine_id, machine_name, device_profile,
                       module, source, location, site_id, zone_id,
                       license_type, license_volume, valid_from, valid_until,
                       rate_limit_rpm, description, created_by)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (company_id, token, hash_, machine_id, machine_name, device_profile,
                     module, source, location, site_id, zone_id,
                     license_type, license_volume, valid_from, valid_until,
                     rate_limit_rpm, description, created_by),
                )
                new_id = cur.lastrowid

        logger.info(f"[ApiKeyService] Created key #{new_id} company={company_id} machine={machine_id} module={module}")
        return {"id": new_id, "token": token, "machine_id": machine_id, "module": module}

    async def validate(self, token: str) -> Optional[ApiKeyInfo]:
        if not xfinder_db.available or not token:
            return None

        hash_ = self.hash_token(token)
        row = await xfinder_db.fetch_one(
            "SELECT * FROM v_vision_api_key_status WHERE hash = %s LIMIT 1",
            (hash_,),
        )
        if not row:
            logger.warning(f"[ApiKeyService] Token not found")
            return None

        info = self._row_to_info(row, token)
        if not info.is_valid:
            logger.warning(f"[ApiKeyService] Token rejected: status={info.status}")
            return None

        import asyncio
        asyncio.create_task(self._register_use(info.id))
        return info

    async def _register_use(self, key_id: int):
        try:
            await xfinder_db.fetch_one(
                "UPDATE vision_api_keys SET last_used_at=NOW(), use_count=use_count+1 WHERE id=%s",
                (key_id,),
            )
        except Exception as e:
            logger.debug(f"[ApiKeyService] Failed to register use: {e}")

    async def list_by_company(self, company_id: int) -> list:
        rows = await xfinder_db.fetch_all(
            """
            SELECT id, company_id, company_name,
                CONCAT(SUBSTR(token,1,8),'...',SUBSTR(token,-4)) AS token_preview,
                machine_id, machine_name, device_profile,
                module, source, location,
                license_type, license_volume,
                valid_from, valid_until, days_remaining,
                status, use_count, last_used_at,
                rate_limit_rpm, active, revoked_at, created_at
            FROM v_vision_api_key_status
            WHERE company_id = %s
            ORDER BY created_at DESC
            """,
            (company_id,),
        )
        return [dict(r) for r in rows]

    async def list_all(self) -> list:
        rows = await xfinder_db.fetch_all(
            """
            SELECT id, company_id, company_name,
                CONCAT(SUBSTR(token,1,8),'...',SUBSTR(token,-4)) AS token_preview,
                machine_id, machine_name, device_profile,
                module, source, location,
                license_type, license_volume,
                valid_from, valid_until, days_remaining,
                status, use_count, last_used_at, active, created_at
            FROM v_vision_api_key_status
            ORDER BY company_id, created_at DESC
            """,
        )
        return [dict(r) for r in rows]

    async def revoke(self, key_id: int, reason: Optional[str] = None) -> bool:
        await xfinder_db.fetch_one(
            "UPDATE vision_api_keys SET active=0, revoked_at=NOW(), revoked_reason=%s WHERE id=%s",
            (reason, key_id),
        )
        logger.info(f"[ApiKeyService] Revoked key #{key_id}")
        return True

    async def toggle_active(self, key_id: int, active: bool) -> bool:
        await xfinder_db.fetch_one(
            "UPDATE vision_api_keys SET active=%s WHERE id=%s",
            (1 if active else 0, key_id),
        )
        return True

    @staticmethod
    def _row_to_info(row: dict, token: str) -> ApiKeyInfo:
        return ApiKeyInfo(
            id=row["id"],
            company_id=row["company_id"],
            company_name=row.get("company_name"),
            token=token,
            machine_id=row["machine_id"],
            machine_name=row.get("machine_name"),
            device_profile=row.get("device_profile"),
            module=row["module"],
            source=row.get("source"),
            location=row.get("location"),
            site_id=row.get("site_id"),
            zone_id=row.get("zone_id"),
            license_type=row.get("license_type"),
            license_volume=int(row.get("license_volume") or 1),
            valid_from=row.get("valid_from"),
            valid_until=row.get("valid_until"),
            days_remaining=row.get("days_remaining"),
            status=row.get("status", "INACTIVE"),
            use_count=int(row.get("use_count") or 0),
            rate_limit_rpm=int(row.get("rate_limit_rpm") or 120),
        )


api_key_service = ApiKeyService()
