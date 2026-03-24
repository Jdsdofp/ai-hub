#!/usr/bin/env python3
import sys
from pathlib import Path

REPO_FILE   = Path("/opt/vision/app/core/repository.py")
ROUTES_FILE = Path("/opt/vision/app/projects/epi_check/api/routes.py")

NEW_METHOD = '''
    async def update_person_presence(
        self,
        company_id: int,
        person_code: str,
        direction: str,
        session_id=None,
    ) -> bool:
        """Atualiza last_entry_at/last_exit_at e is_inside em vision_people."""
        try:
            if direction == "ENTRY":
                await db.execute(
                    """
                    UPDATE vision_people SET
                        last_entry_at      = NOW(),
                        is_inside          = 1,
                        current_session_id = %s,
                        updated_at         = NOW()
                    WHERE company_id = %s AND person_code = %s
                    """,
                    (session_id, company_id, person_code),
                )
            else:
                await db.execute(
                    """
                    UPDATE vision_people SET
                        last_exit_at       = NOW(),
                        is_inside          = 0,
                        current_session_id = NULL,
                        updated_at         = NOW()
                    WHERE company_id = %s AND person_code = %s
                    """,
                    (company_id, person_code),
                )
            logger.info(f"[Repo] update_person_presence: {person_code} {direction}")
            return True
        except Exception as e:
            logger.error(f"[Repo] update_person_presence: {e}")
            return False
'''

ROUTES_OLD_MQTT = '''            # Publica alerta MQTT se negado
            if decision["access_decision"] != "GRANTED":'''

ROUTES_NEW_MQTT = '''            # Atualiza presença em vision_people
            if decision.get("person_code"):
                await repo.update_person_presence(
                    company_id=company_id,
                    person_code=decision["person_code"],
                    direction=session.get("direction", "ENTRY"),
                    session_id=session_id,
                )

            # Publica alerta MQTT se negado
            if decision["access_decision"] != "GRANTED":'''

ROUTES_OLD_CLOSE = '''        return {
            "session_uuid": session_uuid,
            "status": "complete",
            "photos_used": len(photos_list),
            **decision,
        }'''

ROUTES_NEW_CLOSE = '''        if decision.get("person_code"):
            await repo.update_person_presence(
                company_id=company_id,
                person_code=decision["person_code"],
                direction=session.get("direction", "ENTRY"),
                session_id=session["id"],
            )

        return {
            "session_uuid": session_uuid,
            "status": "complete",
            "photos_used": len(photos_list),
            **decision,
        }'''

def patch_file(path, old, new, label):
    text = path.read_text()
    if old not in text:
        print(f"  [SKIP] '{label}' — trecho não encontrado (já aplicado?)")
        return False
    path.write_text(text.replace(old, new, 1))
    print(f"  [OK]   '{label}' aplicado.")
    return True

def insert_method_in_repo(path):
    text = path.read_text()
    marker = "# Singleton — importe nos outros módulos com:"
    if "update_person_presence" in text:
        print("  [SKIP] update_person_presence já existe no repository.py")
        return False
    if marker not in text:
        print("  [ERR]  Marcador não encontrado no repository.py")
        return False
    path.write_text(text.replace(marker, NEW_METHOD + "\n" + marker, 1))
    print("  [OK]   update_person_presence inserido no repository.py")
    return True

print("\n=== fix_presence.py ===\n")
print(f"[1/3] {REPO_FILE}")
insert_method_in_repo(REPO_FILE)
print(f"\n[2/3] {ROUTES_FILE} — validation_photo")
patch_file(ROUTES_FILE, ROUTES_OLD_MQTT, ROUTES_NEW_MQTT, "validation_photo")
print(f"\n[3/3] {ROUTES_FILE} — validation_close")
patch_file(ROUTES_FILE, ROUTES_OLD_CLOSE, ROUTES_NEW_CLOSE, "validation_close")
print("\nPronto! Reinicie: docker restart smartx-vision-v3")
