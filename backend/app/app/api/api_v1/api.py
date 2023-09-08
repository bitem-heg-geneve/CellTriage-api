from fastapi import APIRouter

from app.api.api_v1.endpoints import job
# from app.api.api_v1.endpoints import search

api_router = APIRouter()
api_router.include_router(job.router, prefix="/jobs", tags=["jobs"])
# api_router.include_router(search.router, prefix="/search", tags=["search"])