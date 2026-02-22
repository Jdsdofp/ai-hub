"""
Platform settings loaded from .env file.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    APP_NAME: str = "SmartX Vision Platform"
    APP_VERSION: str = "3.0.0"
    APP_ENV: str = "production"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 5000
    APP_WORKERS: int = 2
    APP_DEBUG: bool = False
    APP_LOG_LEVEL: str = "INFO"
    SECRET_KEY: str = "change-me-in-production"
    API_KEY: str = "smartx-vision-api-key-change-me"

    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "ai_hub"
    MYSQL_POOL_SIZE: int = 10

    MQTT_BROKER: str = "localhost"
    MQTT_PORT: int = 1883
    MQTT_USE_TLS: bool = False
    MQTT_USERNAME: Optional[str] = None
    MQTT_PASSWORD: Optional[str] = None
    MQTT_CLIENT_ID: str = "vision-platform"
    MQTT_TOPIC_PREFIX: str = "smartx/vision"

    DATA_ROOT: str = "./data"

    FACE_MODEL: str = "buffalo_l"
    FACE_DET_SIZE: int = 640
    FACE_SIMILARITY_THRESHOLD: float = 0.45

    EPI_CONFIDENCE_THRESHOLD: float = 0.4
    EPI_INPUT_SIZE: int = 640

    GPU_ENABLED: bool = True
    GPU_DEVICE: int = 0

    EDGE_MODE: bool = False
    EDGE_DEVICE_ID: str = "edge-001"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
