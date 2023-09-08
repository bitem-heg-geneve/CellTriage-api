from typing import Union

from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.job import Job
from app.schemas.job import JobCreate, JobUpdate


class CRUDJob(CRUDBase[Job, JobCreate, JobUpdate]):
    pass


job = CRUDJob(Job)
