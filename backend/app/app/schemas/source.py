from pydantic import BaseModel, HttpUrl, validator
from typing import Optional, List

from typing import Sequence
import app.models.source as model_source
from app.schemas.entity import Entity
from app.schemas.company import Company


class SourceBase(BaseModel):
    class Meta:
        orm_model = model_source.Source


class SourceCreate(SourceBase):
    url: HttpUrl
    text: Optional[str] = None


class SourceUpdate(SourceBase):
    text: Optional[str] = None
    impaakt_score: Optional[float] = None


# Properties shared by models stored in DB
class SourceInDBBase(SourceBase):
    url: HttpUrl
    impaakt_score: Optional[float] = None

    class Config:
        orm_mode = True

    @validator("impaakt_score")
    def is_check(cls, v):
        if v:
            return round(v, 2)
        else:
            return v


# Properties to return to client
class Source(SourceInDBBase):
    text: Optional[str] = None
    NER: List[Entity] = []
    companies: List[Company] = []


# Properties properties stored in DB
class SoureInDB(SourceInDBBase):
    id: int
