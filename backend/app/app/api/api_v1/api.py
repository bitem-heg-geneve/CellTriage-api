from fastapi import APIRouter

from app.api.api_v1.endpoints import job

api_router = APIRouter()
api_router.include_router(job.router, prefix="/jobs", tags=["jobs"])
