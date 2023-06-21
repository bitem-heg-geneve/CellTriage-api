from typing import Union

from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.company import Company
from app.schemas.company import CompanyCreate, CompanyUpdate


class CRUDEntity(CRUDBase[Company, CompanyCreate, CompanyUpdate]):
    pass


company = CRUDEntity(Company)
