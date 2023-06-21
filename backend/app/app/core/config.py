import pathlib

from pydantic import AnyHttpUrl, BaseSettings, EmailStr, validator
from typing import List, Optional, Union
from kombu import Queue
import os


# Project Directories
ROOT = pathlib.Path(__file__).resolve().parent.parent


def route_task(name, args, kwargs, options, task=None, **kw):
    if ":" in name:
        queue, _ = name.split(":")
        return {"queue": queue}
    return {"queue": "ingress"}


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    JWT_SECRET: str = "TEST_SECRET_DO_NOT_USE_IN_PROD"
    ALGORITHM: str = "HS256"

    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8

    # BACKEND_CORS_ORIGINS is a JSON-formatted list of origins
    # e.g: '["http://localhost", "http://localhost:4200", "http://localhost:3000", \
    # "http://localhost:8080", "http://local.dockertoolbox.tiangolo.com"]'
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",
        "http://localhost:8001",  # type: ignore
    ]

    # Origins that match this regex OR are in the above list are allowed
    BACKEND_CORS_ORIGIN_REGEX: Optional[
        str
    ] = "https.*\.(netlify.app|herokuapp.com)"  # noqa: W605

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    SQLALCHEMY_DATABASE_URI: str = os.environ.get("DATABASE_URL", "")
    FIRST_SUPERUSER: EmailStr = "admin@triagecentral.com"
    FIRST_SUPERUSER_PW: str = "secret"

    CELERY_BROKER_URL: str = os.environ.get(
        "CELERY_BROKER_URL", "redis://127.0.0.1:6379/0"
    )
    CELERY_RESULT_BACKEND: str = os.environ.get(
        "CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/0"
    )

    CELERY_TASK_DEFAULT_QUEUE = "default"

    # Force all queues to be explicitly listed in `CELERY_TASK_QUEUES` to help prevent typos
    CELERY_TASK_CREATE_MISSING_QUEUES = False

    CELERY_TASK_QUEUES: list = (
        # need to define default queue here or exception would be raised
        Queue("default"),
        Queue("ingress"),
        Queue("torch"),
    )

    CELERY_TASK_ROUTES = (route_task,)

    class Config:
        case_sensitive = True


settings = Settings()
