from pydantic import BaseModel, HttpUrl, validator
from typing import Optional

from typing import Sequence
import app.models.company as model_company


class CompanyBase(BaseModel):
    class Meta:
        orm_model = model_company.Company


class CompanyCreate(CompanyBase):
    source_id: int
    company: str
    score: float


class CompanyUpdate(CompanyBase):
    company: Optional[str] = None
    score: Optional[float] = None


# Properties shared by models stored in DB
class CompanyInDBBase(CompanyBase):
    company: str
    score: float

    class Config:
        orm_mode = True


# Properties to return to client
class Company(CompanyInDBBase):
    pass


# Properties properties stored in DB
class CompanyInDB(CompanyInDBBase):
    id: int
