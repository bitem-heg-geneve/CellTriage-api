from typing import Union

from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.source import Source
from app.schemas.source import SourceCreate, SourceUpdate


class CRUDSource(CRUDBase[Source, SourceCreate, SourceUpdate]):
    pass


source = CRUDSource(Source)
