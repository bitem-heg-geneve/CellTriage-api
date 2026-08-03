from pydantic import BaseModel, HttpUrl, Field, validator

from typing import Sequence, List, Union
from datetime import datetime
from app.schemas.article import Article
from fastapi import Query
from typing import Optional, Dict
from datetime import datetime

from app.schemas.article import ArticleCreate, ArticleUpdate
from pydantic import root_validator


class JobBase(BaseModel):
    pass


class JobCreate(JobBase):
    use_fulltext: bool = Field(default=False)
    article_set: List[ArticleCreate] = Field(default=[])
    
    class Config:
        schema_extra = {
            "example": {
                "use_fulltext": False,
                "article_set": [
                    {"pmid": 41892333},  # HeLa cell line - HIGH score
                    {"pmid": 41888353},  # CHO cell line - HIGH score
                    {"pmid": 41890120},  # rhesus iPSCs - HIGH score
                    {"pmid": 41895740},  # Mouse pancreatic - HIGH score
                    {"pmid": 41890579},  # Postoperative pain - LOW score
                    {"pmid": 41888632},  # Smell/taste disorder - LOW score
                    {"pmid": 41890924},  # Surgical psychology - LOW score
                    {"pmid": 41888764},  # Malpractice fear - LOW score
                ]
            }
        }


class JobUpdate(JobBase):
    status: Optional[str] = None
    process_start_at: Optional[datetime] = None
    process_end_at: Optional[datetime] = None


# Properties shared by models stored in DB
class JobInDBBase(JobBase):
    id: int
    use_fulltext: bool = Field(default=True)
    status: str = Field(default="pending")
    job_created_at: datetime = Field(default=datetime.now())
    process_start_at: Optional[datetime] = None
    process_end_at: Optional[datetime] = None
    process_time: int = 0

    class Config:
        orm_mode = True

    @root_validator
    def compute_process_time(cls, values) -> Dict:
        process_start_at = values.get("process_start_at")
        process_end_at = values.get("process_end_at")
        if process_start_at:
            if process_end_at:
                values["process_time"] = round(
                    (process_end_at - process_start_at).total_seconds(), 2
                )
            else:
                values["process_time"] = round(
                    (datetime.now() - process_start_at).total_seconds(), 2
                )
        else:
            values["process_time"] = 0.00
        return values


# Properties to return to client
class Job(JobInDBBase):
    article_set: List[Article]


# Status to return to client
class JobStatus(JobInDBBase):
    pass


# Properties properties stored in DB
class JobInDB(JobInDBBase):
    process_start_at: Optional[datetime] = None
    process_end_at: Optional[datetime] = None
    pass
