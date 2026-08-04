from pydantic import BaseModel, HttpUrl, Field, model_validator

from typing import Sequence, List, Union
from datetime import datetime
from app.schemas.article import Article
from fastapi import Query
from typing import Optional, Dict
from datetime import datetime

from app.schemas.article import ArticleCreate, ArticleUpdate


class JobBase(BaseModel):
    pass


class JobCreate(JobBase):
    use_fulltext: bool = Field(default=False)
    article_set: List[ArticleCreate] = Field(default=[])

    model_config = {
        "json_schema_extra": {
            "example": {
                "use_fulltext": False,
                "article_set": [
                    {"pmid": 41892333},
                    {"pmid": 41888353},
                    {"pmid": 41890120},
                    {"pmid": 41895740},
                    {"pmid": 41890579},
                    {"pmid": 41888632},
                    {"pmid": 41890924},
                    {"pmid": 41888764},
                ]
            }
        }
    }


class JobUpdate(JobBase):
    status: Optional[str] = None
    process_start_at: Optional[datetime] = None
    process_end_at: Optional[datetime] = None


class JobInDBBase(JobBase):
    id: int
    use_fulltext: bool = Field(default=True)
    status: str = Field(default="pending")
    job_created_at: datetime = Field(default_factory=datetime.now)
    process_start_at: Optional[datetime] = None
    process_end_at: Optional[datetime] = None
    process_time: float = 0

    model_config = {"from_attributes": True}

    @model_validator(mode="after")
    def compute_process_time(self) -> "JobInDBBase":
        if self.process_start_at:
            end = self.process_end_at or datetime.now()
            self.process_time = round(
                (end - self.process_start_at).total_seconds(), 2
            )
        else:
            self.process_time = 0.0
        return self


class Job(JobInDBBase):
    article_set: List[Article]


class JobStatus(JobInDBBase):
    pass


class JobInDB(JobInDBBase):
    process_start_at: Optional[datetime] = None
    process_end_at: Optional[datetime] = None
