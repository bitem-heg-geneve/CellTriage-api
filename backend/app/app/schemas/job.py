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
    use_fulltext: bool = Field(default=True)
    article_set: List[ArticleCreate] = Field(
        example=[
            # ArticleCreate(pmid=14691011),
            # ArticleCreate(pmid=25190367),
            ArticleCreate(pmid=36585756),
            ArticleCreate(pmid=36564873),
            ArticleCreate(pmid=35985809),
            ArticleCreate(pmid=34915666),
            ArticleCreate(pmid=35183060),
            ArticleCreate(pmid=10390151),
            ArticleCreate(pmid=31654625),
            ArticleCreate(pmid=31678775),
            ArticleCreate(pmid=31741260),
            ArticleCreate(pmid=32289117),
            
            
        ],
        default=[],
    )


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

    # @validator("process_time")
    # def pt_check(cls, v):
    #     return round(v, 2)


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
