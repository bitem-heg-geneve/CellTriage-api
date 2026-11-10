import pathlib

from pydantic import AnyHttpUrl, EmailStr, field_validator
from pydantic_settings import BaseSettings
from typing import List, Optional, Union
from kombu import Queue
import os

from app.services.sibils_api import sibils_base_url


ROOT = pathlib.Path(__file__).resolve().parent.parent


def route_task(name, args, kwargs, options, task=None, **kw):
    if ":" in name:
        queue, _ = name.split(":")
        return {"queue": queue}
    return {"queue": "default"}


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    JWT_SECRET: str = "TEST_SECRET_DO_NOT_USE_IN_PROD"
    ALGORITHM: str = "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8

    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",
        "http://localhost:8001",
    ]

    BACKEND_CORS_ORIGIN_REGEX: Optional[str] = "https.*\\.(netlify.app|herokuapp.com)"

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    SQLALCHEMY_DATABASE_URI: str = os.environ.get("DATABASE_URL", "")
    FIRST_SUPERUSER: EmailStr = "admin@triagecentral.com"
    FIRST_SUPERUSER_PW: str = "secret"

    CELERY_BROKER_URL: str = os.environ.get("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0")
    CELERY_RESULT_BACKEND: str = os.environ.get("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/0")
    CELERY_BROKER_HEARTBEAT: int = 0
    CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP: bool = True
    CELERY_BROKER_TRANSPORT_OPTIONS: dict = {"visibility_timeout": 14400}
    CELERY_TASK_ACKS_LATE: bool = True
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = 1
    CELERY_TASK_DEFAULT_QUEUE: str = "default"
    CELERY_TASK_CREATE_MISSING_QUEUES: bool = False

    CELERY_TASK_QUEUES: list = (
        Queue("default"),
        Queue("ingress"),
        Queue("infer"),
    )

    CELERY_TASK_ROUTES: tuple = (route_task,)

    # Shared triage guard (2026-11-10). SIBILS_URL is required, no default on
    # purpose: the text source is part of the deployment, like the model. Base
    # URL, http(s); a trailing slash is tolerated and stripped (the client
    # appends /fetch). Checked below, because pydantic does not validate defaults.
    SIBILS_URL: str = ""
    SIBILS_TIMEOUT: int = 30   # seconds, connect and read, per request
    SIBILS_RETRIES: int = 5    # attempts per batch (backoff 2 s -> 30 s) before fetch_error
    # app.readiness: minimum score of the positive control; 0 skips the SIBiLS
    # fetch part of the gate (offline development only).
    CELLTRIAGE_READINESS_MIN: float = 0.6
    CELLTRIAGE_READINESS_FETCH: bool = True

    model_config = {"case_sensitive": True}


settings = Settings()
settings.SIBILS_URL = sibils_base_url(settings.SIBILS_URL)
