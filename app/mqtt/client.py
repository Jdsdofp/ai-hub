"""
MQTT Client for SmartX HUB integration.
Topics: smartx/vision/{company_id}/{event_type}
"""
import json
import time
from loguru import logger
from app.core.config import settings


class MQTTClient:
    def __init__(self):
        self._connected = False

    async def connect(self):
        try:
            import aiomqtt
            self._connected = True
            logger.info(f"MQTT ready: {settings.MQTT_BROKER}:{settings.MQTT_PORT}")
        except Exception as e:
            logger.warning(f"MQTT init failed: {e}. Running without MQTT.")
            self._connected = False

    async def disconnect(self):
        self._connected = False

    def _topic(self, company_id: int, suffix: str) -> str:
        return f"{settings.MQTT_TOPIC_PREFIX}/{company_id}/{suffix}"

    async def publish(self, company_id: int, topic_suffix: str, payload: dict):
        if not self._connected:
            return
        try:
            import aiomqtt
            async with aiomqtt.Client(
                hostname=settings.MQTT_BROKER, port=settings.MQTT_PORT,
                username=settings.MQTT_USERNAME, password=settings.MQTT_PASSWORD,
            ) as client:
                await client.publish(self._topic(company_id, topic_suffix), json.dumps(payload))
        except Exception as e:
            logger.error(f"MQTT publish error: {e}")

    async def publish_detection(self, company_id: int, result: dict):
        payload = {
            "event": "detection", "company_id": company_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "compliant": result.get("compliant", False),
            "missing": result.get("missing", []),
            "faces": result.get("faces", []),
            "source": settings.EDGE_DEVICE_ID if settings.EDGE_MODE else "server",
        }
        await self.publish(company_id, "detection", payload)

    async def publish_alert(self, company_id: int, alert_type: str, details: dict):
        payload = {
            "event": "alert", "company_id": company_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "alert_type": alert_type, **details,
        }
        await self.publish(company_id, "alert", payload)

    @property
    def is_connected(self):
        return self._connected


mqtt_client = MQTTClient()
